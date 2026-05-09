# OMNI-LOG — Theoretical Background

This document explains the core ML concepts used in OMNI-LOG, mapped to the specific design choices made in this system.

---

## 1. BIO Tagging (Named Entity Recognition Scheme)

**What it is:**
BIO (Beginning–Inside–Outside) is a labeling scheme used in sequence labeling tasks. Each token in a sequence receives one of three prefixes:
- `B-` (Beginning): the first token of an entity span
- `I-` (Inside): a continuation token within the same span
- `O` (Outside): not part of any named entity

**Why it matters:**
BIO tagging turns log parsing into a standard NLP sequence labeling problem. Instead of writing hand-crafted regex for every log format, the model learns to recognize temporal, structural, and semantic patterns from data.

**OMNI-LOG's tag set:**

```
<PAD>  O  B-TIME  I-TIME  B-LEVEL  I-LEVEL  B-COMPONENT  I-COMPONENT  B-PARAM  I-PARAM
```

10 classes total. The model outputs one tag per token; together the tag sequence describes the full semantic structure of a log line.

**Example:**
```
Tokens: 2015-10-18  18:01:47  INFO  NameSystem  completeFile  blk_-160
Tags:   B-TIME      I-TIME    B-LEVEL B-COMPONENT O             B-PARAM
```

---

## 2. BiLSTM — Bidirectional Long Short-Term Memory

**What it is:**
An LSTM (Long Short-Term Memory) is a recurrent neural network cell designed to capture long-range dependencies in sequences. A standard LSTM reads tokens left-to-right; a **bidirectional** LSTM (BiLSTM) runs two separate LSTMs in parallel — one forward, one backward — and concatenates their hidden states at each position.

**Architecture:**
```
Token[1..N]  →  Embedding Layer  →  BiLSTM  →  Hidden States[1..N]
                                   ┌──────┐
                   Forward LSTM:   → → → →
                   Backward LSTM:  ← ← ← ←
                                   └──────┘
                   Concatenated output: [forward_h ; backward_h]
```

**OMNI-LOG configuration:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `embedding_dim` | 128 | Token embedding size |
| `hidden_dim` | 256 | Total hidden state (128 per direction) |
| `num_layers` | 1 | Single-layer BiLSTM |
| `dropout` | 0.1 | Regularization |
| `batch_first` | True | Input shape: `[batch, seq, feature]` |

**Why BiLSTM over standard LSTM:**
Log parsing requires both left and right context. Whether `INFO` is a log level or part of a filename depends on surrounding tokens. The backward LSTM sees right-side context that the forward LSTM cannot.

**Defined in:** `src/features/chunker/model.py` — `BiLSTM_CRF._get_lstm_features()`

---

## 3. CRF — Conditional Random Field

**What it is:**
A CRF (Conditional Random Field) is a probabilistic graphical model that finds the **globally optimal** label sequence for a given input, considering label-to-label transition scores alongside per-token emission scores from the BiLSTM.

**Why it is used after BiLSTM:**
Without a CRF, the BiLSTM produces independent per-token probability distributions. This can lead to structurally invalid tag sequences (e.g., `I-TIME` after `B-PARAM` — an impossible transition). The CRF layer learns a transition matrix that penalizes invalid tag transitions.

**Forward algorithm (training):** computes the log-partition function — the sum of scores over all possible tag sequences.

**Viterbi algorithm (inference):** dynamic programming search that finds the single highest-scoring valid tag sequence.

**OMNI-LOG implementation:**
- Library: `TorchCRF`
- Training: `model(tokens, tags, mask)` returns negative log-likelihood
- Inference: `model.decode(tokens, mask)` returns the Viterbi-decoded tag sequence

**Defined in:** `src/features/chunker/model.py` — `BiLSTM_CRF.forward()` (training), `BiLSTM_CRF.decode()` (inference)

---

## 4. Siamese Network

**What it is:**
A Siamese Network is a neural architecture in which **two (or more) identical sub-networks share the same weights**. They each process a different input and produce embeddings that are then compared. Because the weights are shared, the network learns a single distance function that is consistent across all inputs.

**Original motivation (Bromley et al., 1994):** signature verification — compare two handwritten signatures to determine if they came from the same person.

**OMNI-LOG adaptation:**
The Siamese Network compares log lines against known templates. Both the log and the template are passed through the same `LogEncoder`; if their embeddings are close in vector space, the log matches the template.

```
Raw Log ──┐
           ├── LogEncoder (shared weights) ──→ vec_a ──┐
Template ──┘                                           ├── Cosine Similarity
                                                       └── vec_b
```

**SiameseNet** in `src/features/siamese/encoder.py` wraps the shared `LogEncoder` and computes dot-product similarity between L2-normalized vectors (equivalent to cosine similarity).

---

## 5. Triplet Loss

**What it is:**
Triplet loss trains an embedding network by presenting three examples simultaneously:
- **Anchor (A):** a raw log line
- **Positive (P):** the correct template for that log (semantically similar)
- **Negative (N):** a randomly selected wrong template (semantically different)

The loss pushes the anchor closer to the positive and farther from the negative in embedding space:

```
L = max(0,  d(A, P)  −  d(A, N)  +  margin )
```

Where `d` is Euclidean distance and `margin` is a minimum required separation (set to `1.0` in OMNI-LOG).

**Intuition:** the network learns that `"Receiving block blk_-160"` should be near `"Receiving block <*>"` (positive) and far from `"File not found: <*>"` (negative).

**OMNI-LOG implementation:**
- Loss function: `torch.nn.TripletMarginLoss(margin=1.0, p=2)`
- Negative mining: **random negative sampling** — any template that is not the correct one for the anchor
- Defined in: `src/features/siamese/train_siamese.py`

---

## 6. Masked Mean Pooling

**What it is:**
After the BiLSTM produces a hidden state for each token, the Siamese encoder needs a **single fixed-size vector** to represent the entire log. Masked mean pooling averages only the real (non-padding) token states:

```
pooled = sum(hidden_states × mask) / sum(mask)
```

**Why masking matters:**
Without masking, padding zeros would dilute the average and make logs of different lengths non-comparable. Masked pooling ensures the summary vector reflects only genuine content.

**Defined in:** `src/features/siamese/encoder.py` — `LogEncoder.forward()`

---

## 7. L2 Normalization

After pooling, the Siamese encoder applies L2 normalization to each embedding vector:

```python
F.normalize(pooled_output, p=2, dim=1)
```

This projects all vectors onto a unit hypersphere. The key consequence: **cosine similarity reduces to a dot product** — computationally cheaper and stable for nearest-neighbor search across large template libraries.

**Defined in:** `src/features/siamese/encoder.py` — `LogEncoder.forward()`

---

## 8. Confidence-Based Hybrid Routing

**What it is:**
OMNI-LOG's inference pipeline uses a **two-stage hybrid architecture**:

1. **Chunker (BiLSTM-CRF):** fast primary parser. Computes per-token softmax probabilities and averages them to produce a confidence score in `[0, 1]`.
2. **Siamese Resolver (fallback):** if confidence < 0.90, the raw log is encoded and compared against the template library via cosine similarity (threshold 0.85).

```
Log → Chunker → confidence ≥ 0.90 → SUCCESS (use chunker tags)
                             < 0.90 → Siamese Resolver → RESOLVED / UNKNOWN
```

This routing prevents the chunker from silently producing low-quality parses on unseen or unusual log patterns.

**Defined in:** `src/features/engine/pipeline.py`, `src/features/siamese/resolver.py`

---

## 9. Train / Validation Split (80/20)

**What it is:**
The complete labeled dataset is randomly divided into:
- **Training set (80%):** used to compute gradients and update model weights
- **Validation set (20%):** held out during training; used to measure generalization after each epoch

**Purpose of the split:**
The validation loss reveals whether the model is generalizing or memorizing training examples. In OMNI-LOG, validation loss plateaus at ~0.50 after epoch 3, which is the signal for selecting the optimal checkpoint (early stopping around epoch 5–6).

**OMNI-LOG implementation:**
```python
train_size = int(0.8 * len(full_dataset))  # 80%
val_size = len(full_dataset) - train_size  # 20%
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
```

**Reproducibility:** `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` are all set at the top of the training script for deterministic splits across runs.

**Defined in:** `src/features/chunker/train.py` line 43–48

---

## 10. Adam Optimizer with Gradient Clipping

**Adam (Adaptive Moment Estimation):**
Adam adjusts the learning rate per parameter by tracking the first moment (mean) and second moment (variance) of gradients. It is the standard optimizer for NLP models because it converges faster than SGD on sparse gradient problems.

**Configuration:** `lr=0.001`, default betas `(0.9, 0.999)`, epsilon `1e-8`

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```
Scales down the gradient vector whenever its L2 norm exceeds 5.0. This prevents "exploding gradients" — a common failure mode in RNN training where large gradients destabilize weight updates.

---

## 11. MLflow Experiment Tracking

OMNI-LOG logs all training runs to MLflow:
- **Chunker experiment:** `OMNI-LOG-Chunker` — logs `train_loss` and `val_loss` per epoch
- **Siamese experiment:** `OMNI-LOG-Siamese` — logs `siamese_loss` per epoch
- Hyperparameters (`embedding_dim`, `hidden_dim`, `lr`, `epochs`, `batch_size`) are logged as run parameters
- The best model checkpoint is logged as an artifact

**Run artifacts directory:** `mlruns/` (auto-created in the project root)

---

## 12. Viterbi Decoding

**What it is:**
The Viterbi algorithm is a dynamic programming method that finds the single most probable tag sequence across all possible sequences. It is used at inference time by the CRF layer.

**Complexity:** O(T × K²) where T = sequence length and K = number of tags. With T=64 and K=10, this is trivially fast.

**OMNI-LOG call:** `model.crf.viterbi_decode(features, mask)` — returns the optimal tag path per log in the batch.

---

## Architecture Summary

```
                   OMNI-LOG Architecture
                   ─────────────────────

Input Log String
      │
      ▼
 [Tokenizer]           regex-based, domain-agnostic
      │
      ▼
 [Embedding Layer]     token → 128-dim dense vector
      │
      ▼
 [BiLSTM]              128-dim input → 256-dim output (bidirectional)
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
 [CRF + Viterbi]               [Masked Mean Pool]
  ↓ BIO tag sequence             ↓ 256-dim log vector
  ↓ + confidence score           ↓
      │                          ▼
      │               [L2 Normalize → Cosine Sim]
      │                          ↓ template match score
      │
      └──── Hybrid Router ───────┘
            (confidence ≥ 0.90 → chunker output)
            (confidence < 0.90 → siamese fallback)
                  │
                  ▼
          Structured JSON Output
```

---

## References

1. **BIO Tagging / NER:**
   Ramshaw, L. A., & Marcus, M. P. (1995). Text chunking using transformation-based learning. *Proceedings of the 3rd ACL Workshop on Very Large Corpora.*

2. **Long Short-Term Memory (LSTM):**
   Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

3. **Bidirectional RNNs:**
   Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

4. **CRF for Sequence Labeling:**
   Lafferty, J., McCallum, A., & Pereira, F. C. N. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML 2001.*

5. **BiLSTM-CRF for NER:**
   Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT 2016.*

6. **Siamese Networks:**
   Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1994). Signature verification using a Siamese time delay neural network. *NIPS 1994.*

7. **Triplet Loss / Metric Learning:**
   Hoffer, E., & Ailon, N. (2015). Deep Metric Learning Using Triplet Network. *ICLR Workshop 2015.*
   Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *CVPR 2015.*

8. **Adam Optimizer:**
   Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015.*

9. **LogHub Dataset (source of *_2k.log files):**
   He, P., Zhu, J., He, S., Li, J., & Lyu, M. R. (2020). An Evaluation Study on Log Parsing and Its Use in Log Mining. *IEEE/IFIP DSN 2016.*
   Zhu, J., He, S., Liu, J., He, P., Xia, Q., Zheng, Z., & Lyu, M. R. (2019). Tools and Benchmarks for Automated Log Parsing. *ICSE-SEIP 2019.*

10. **Log Parsing / Automated Log Analysis:**
    He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). Drain: An Online Log Parsing Approach with Fixed Depth Tree. *IEEE ICWS 2017.*
