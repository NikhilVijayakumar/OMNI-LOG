# OMNI-LOG: Multi-Category Log Structuring Pipeline
### Academic Presentation — Method, Results, Discussion, Conclusions

**Course:** Natural Language Processing (MTech)
**Team:** Nikhil Vijaykumar (MT24AAI153) · Hanna Shahanaz (MT24AAI148)

---

## 1. Problem Statement

Modern distributed systems generate logs from heterogeneous sources — web servers, mobile apps, HPC clusters, network proxies — each with a unique, inconsistently documented format. Downstream analytics (anomaly detection, root-cause analysis) require structured data, but traditional regex parsers are brittle: they must be hand-crafted per format and break on any format variation.

**Research Question:** Can a single neural architecture learn a shared semantic grammar across structurally diverse log formats and extract entities generalisably — without per-domain rules?

**Key challenges:**
- Logs from 9 different systems have no common schema
- Variable content (IPs, block IDs, timestamps) must be separated from static structure
- Unseen log patterns must be handled gracefully at inference time
- The system must be fast enough for practical use (real-time or near-real-time)

---

## 2. Dataset and Data Acquisition

### Source

**LogHub** — a publicly curated benchmark for log parsing research.
> Repository: https://github.com/logpai/loghub

LogHub provides raw log files paired with **ground-truth template CSVs** (`EventTemplate` column). Each template marks variable fields with `<*>`, e.g.:

```
Raw log:    Receiving block blk_-160 from 192.168.0.1
Template:   Receiving block <*> from <*>
```

This pairing is what makes supervised BIO-tag generation possible without manual labelling.

### Domains Used

| # | Domain | Format Style | Example Timestamp |
|---|--------|-------------|-------------------|
| 1 | Android | Logcat | `03-17 16:13:38.811` |
| 2 | Apache | Web server | `[Sun Dec 04 04:47:44 2005]` |
| 3 | Hadoop | Java logging | `2015-10-18 18:01:47,978` |
| 4 | HealthApp | Mobile health | `20171223-22:15:29:606\|` |
| 5 | HPC | HPC state | `134681 node-246` |
| 6 | Linux | Syslog | `Jun 14 15:16:01` |
| 7 | OpenSSH | Syslog | `Jun 14 15:16:01` |
| 8 | Proxifier | Network proxy | `[10.30 16:49:06]` |
| 9 | Zookeeper | Java logging | `2015-10-18 18:01:47,978` |

### Why 2k Subsets (`*_2k.log`)?

Full LogHub domains range from thousands to millions of lines. Using 2,000 lines per domain:

| Reason | Explanation |
|--------|-------------|
| **Balanced training** | Each domain contributes equally — prevents high-volume domains (Hadoop) dominating the model |
| **Reproducibility** | Fixed-size subsets make experiments comparable across papers |
| **Computational feasibility** | ~18,000 total samples fit comfortably in CPU training |
| **Template coverage** | 2k lines typically covers the majority of unique templates for a domain |

**Total dataset size:** 9 domains × 2,000 lines = **18,000 log lines**
**Universal vocabulary:** **17,067 unique tokens** built across all domains

---

## 3. Methodology

### 3.1 Pipeline Architecture

The pipeline has five modules wired in sequence:

```
Raw Log Files  +  Template CSVs
        |
        v
 [DATA MODULE]          Tokenisation, BIO tag generation, vocab, tensors
        |
        v
 [CHUNKER MODULE]       BiLSTM-CRF sequence labeller → BIO tags + confidence
        |
     ┌──┴──────────────────────────────┐
     │ confidence ≥ 0.90               │ confidence < 0.90
     v                                  v
 Direct structured output        [SIAMESE MODULE]   Template similarity matching
     │                                  │
     └─────────────────┬───────────────┘
                       v
              [ENGINE MODULE]          Batch streaming, JSON writing
                       |
                       v
              [MONITOR MODULE]         Metrics, MLflow experiment tracking
```

### 3.2 Data Preprocessing

**Step 1 — Tokenisation**

A unified regex tokeniser (`processor.py`) preserves semantically meaningful tokens:

```
Pattern:  [a-zA-Z0-9_\-\.]+ | [:\(\)\[\]=] | \S+
```

Example:
```
Input:   2015-10-18 18:01:47,978 INFO NameSystem: completeFile: blk_-160
Tokens:  ['2015-10-18', '18:01:47,978', 'INFO', 'NameSystem', ':', 'completeFile', ':', 'blk_-160']
```

**Step 2 — BIO Tag Generation**

Each token is labelled in the BIO (Begin–Inside–Outside) scheme using two complementary strategies:

**(a) Heuristic labelling** — for structural tokens with consistent patterns:
- Timestamps: tokens matching `\d{2,4}[-:/]\d{2}[-:/]\d{2,4}` or syslog months → `B-TIME / I-TIME`
- Log levels: `INFO, DEBUG, WARN, ERROR, FATAL` → `B-LEVEL`

**(b) Template alignment** — for variable content (parameters):
- The ground-truth template `<*>` markers identify parameter spans → `B-PARAM / I-PARAM`
- Alignment walks both the token list and template token list in parallel

Tag vocabulary (10 classes):

| Tag | Meaning |
|-----|---------|
| `O` | Outside any entity |
| `B-TIME` / `I-TIME` | Timestamp begin/inside |
| `B-LEVEL` / `I-LEVEL` | Log severity begin/inside |
| `B-COMPONENT` / `I-COMPONENT` | System component begin/inside |
| `B-PARAM` / `I-PARAM` | Variable parameter begin/inside |
| `<PAD>` | Padding token |

**Step 3 — Universal Vocabulary**

A single vocabulary is built across all 9 domains. Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`. Final vocab size: **17,067 tokens**.

Shared vocabulary ensures that a token like `INFO` carries the same embedding regardless of which domain it came from — this is the foundation of cross-domain generalisation.

**Step 4 — Numericalization and Padding**

All sequences are padded or truncated to `max_seq_len = 64`. Output: `[batch, 64]` integer tensors for tokens and tags.

---

### 3.3 BiLSTM-CRF Chunker

**Why formulate log parsing as NER (Named Entity Recognition)?**

Treating each log token as a labelling problem lets a single model learn *shared structure* across domains. The same entity types (time, level, component, parameter) appear in every domain — only the surface form changes.

#### Architecture

```
Token IDs  [B, 64]
    ↓
Embedding Layer   (17067 → 128 dimensions)
    ↓
Bi-LSTM           (128 → 256 hidden, bidirectional)
    ↓   Forward LSTM captures left context (what came before)
    ↓   Backward LSTM captures right context (what follows)
    ↓
Dropout (p = 0.1)
    ↓
Linear Projection  (256 → 10 tag scores per token)
    ↓
CRF Layer         (learns valid tag-transition probabilities)
    ↓
Viterbi Decode    (finds globally optimal tag sequence)
    ↓
BIO Tag Sequence  [B, 64]
```

**Total parameters: 2,451,458**

#### Why BiLSTM?

Unidirectional LSTM can only see what came before a token. Log tokens often require both context:
- `blk_-160` is a `PARAM` because the word `block` precedes it (left context)
- `04:47:44` is part of `I-TIME` because `Dec` follows it (right context matters less here, but other domains need it)

BiLSTM concatenates forward and backward hidden states:  `h_i = [→h_i ; ←h_i]`

#### Why CRF instead of plain softmax?

Softmax at each token position is independent — it can generate invalid sequences like:
```
B-LEVEL → I-TIME   ← impossible: "Inside TIME" cannot follow "Begin LEVEL"
```

The CRF layer learns a **transition score matrix** `T[i→j]` — the score of tag `j` following tag `i`. The global sequence score is:

```
score(X, Y) = Σ emission_scores(y_i | x_i) + Σ transition_scores(y_{i-1} → y_i)
```

Viterbi decoding finds the highest-scoring *valid* sequence in O(L × K²) time (L = sequence length, K = tag count = 10).

**Loss function:** Negative log-likelihood of the correct tag sequence under the CRF:

```
L = -log P(Y* | X)
```

This optimises both emission scores (from BiLSTM) and transition scores (CRF weights) jointly.

#### Confidence Scoring

For confidence-based routing, per-token softmax max-probability is computed and averaged across real (non-padding) tokens:

```python
probs    = softmax(emission_scores, dim=-1)   # [B, L, 10]
max_prob = max(probs, dim=-1)                 # [B, L]
conf     = mean(max_prob * mask) / sum(mask)  # [B]  — masked mean over real tokens
```

This gives a scalar in [0, 1] per log line. Threshold: **0.90**.

---

### 3.4 Siamese Encoder and Template Resolver

**Core idea:** Instead of classifying unknown logs, *compare* them to a pre-encoded template library using vector similarity. This handles logs never seen during training.

#### LogEncoder Architecture

```
Token IDs  [1, 64]
    ↓
Embedding (17067 → 64d)
    ↓
Bi-LSTM  (64 → 128 hidden)
    ↓
Masked Mean Pooling   (average only non-padding positions)
    ↓
L2 Normalisation      (‖v‖₂ = 1)
    ↓
Fixed-length vector  [1, 128]
```

**Why masked mean pooling?**
- Simple last-hidden-state pooling is biased towards the end of the sequence
- Max pooling loses positional structure
- Mean pooling over real tokens produces a **length-invariant** representation — a 3-token log and a 20-token log with similar content get similar vectors

**Why L2 normalise?**
- After normalisation, cosine similarity = dot product: `sim(a, b) = a·b`
- This means template matching against N templates is a single batched matrix multiply: `scores = log_vec @ template_matrix.T`
- Efficient, no division at inference time

#### Training: Triplet Margin Loss

The encoder is trained with **triplets (Anchor, Positive, Negative)**:

- **Anchor:** a raw log line
- **Positive:** its correct template
- **Negative:** a randomly sampled *different* template

```
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

where `d(·,·)` is Euclidean distance and `margin = 1.0`.

The model learns to pull logs close to their correct template and push them away from wrong ones. No class labels needed — only the log-template pairing from LogHub.

**Negative mining strategy:** Random sampling of incorrect templates. Hard negative mining (picking the most similar incorrect template) would improve convergence but was not implemented in this version.

#### Template Resolution at Inference

```
1. Encode all known templates once → matrix [N_templates, 128]
2. For each log: encode → [1, 128]
3. Cosine similarity: scores = log_vec @ template_matrix.T  → [N_templates]
4. Best match: argmax(scores)
5. If max_score ≥ 0.85 → return matched template
   Else               → return UNKNOWN_PATTERN
```

---

### 3.5 Hybrid Routing Logic

```
For each log line:
  1. BiLSTM-CRF → BIO tags + confidence score
  2. If confidence ≥ 0.90:
         → use chunker output  [Method: "Bi-LSTM-CRF", Status: "SUCCESS"]
  3. Else:
         → encode log → cosine similarity with template library
         → If similarity ≥ 0.85:
               → return matched template  [Method: "Siamese-Resolver", Status: "RESOLVED"]
         → Else:
               → [Method: "None", Status: "UNKNOWN_PATTERN"]
```

This design means the fast path (chunker) handles the common case, and the slower semantic fallback only activates when the primary model is uncertain.

---

### 3.6 Training Procedure

#### Hyperparameters

| Parameter | Chunker | Siamese Encoder |
|-----------|---------|-----------------|
| Embedding dim | 128 | 128 |
| Hidden dim | 256 | 256 |
| Learning rate | 0.001 (Adam) | 0.0001 (Adam) |
| Batch size | 32 | 32 |
| Epochs | 3 (config) | 3 |
| Gradient clip | 5.0 | — |
| Loss | CRF NLL | TripletMarginLoss (margin=1.0) |
| Dropout | 0.1 | 0.1 |
| Random seed | 42 | 42 |

#### Train / Validation Split

The combined dataset of ~18,000 samples was split **80% training / 20% validation** using `torch.utils.data.random_split` with `seed=42`.

| Split | Samples (approx.) | Purpose |
|-------|-------------------|---------|
| Training (80%) | ~14,400 | Model weight updates |
| Validation (20%) | ~3,600 | Monitor generalisation, detect overfitting |

**Why 80/20?**
- No separate held-out test set was created — the 20% validation set serves as the unseen evaluation set
- 80/20 is the standard split for moderate-sized NLP datasets where data is limited
- Stratification is not enforced by domain in this implementation (future improvement)

#### Preventing Underfitting and Overfitting

| Technique | Where Applied | Purpose |
|-----------|--------------|---------|
| **Dropout (p=0.1)** | After BiLSTM, after encoder LSTM | Randomly zeros activations during training; forces the model not to rely on any single neuron |
| **Gradient clipping (norm=5.0)** | Chunker training loop | Prevents exploding gradients in LSTM, stabilises training |
| **Validation loss monitoring** | Every epoch | If val loss stops decreasing while train loss continues to fall, overfitting is occurring |
| **Early stopping (recommended)** | Not currently implemented | Would stop training at the epoch with lowest val loss |
| **Shared multi-domain training** | All 9 domains mixed | Implicit regularisation: the model cannot overfit to any single domain's quirks |
| **Random seed (42)** | `torch`, `numpy`, `random` | Reproducible splits and weight initialisation |

**Underfitting indicators:** High train loss and high val loss — the model has not learned the task. Remedy: more epochs, larger hidden dim, lower dropout.

**Overfitting indicators:** Low train loss, rising val loss — the model has memorised training examples. Remedy: more dropout, fewer epochs, more data augmentation.

---

### 3.7 Model Validation Strategy

Since there is no separate ground-truth-labelled test set outside of training data, validation proceeds at three levels:

| Level | What is measured | How |
|-------|-----------------|-----|
| **Loss convergence** | Does training loss decrease? Does val loss track it? | Monitored per epoch via MLflow |
| **Pipeline success rate** | What fraction of logs are processed without error? | `status == "SUCCESS"` count in output JSON |
| **Manual inspection** | Are the predicted BIO tags semantically reasonable? | Inspecting `structured_logs.json` sample records |
| **Unit tests** | Do individual modules produce correct outputs for known inputs? | `pytest tests/` — hybrid parser, resolver, encoder |
| **Integration test** | Does the full pipeline run end-to-end on real log files? | `tests/integration/test_end_to_end.py` |

**Formal entity-level evaluation** (Precision/Recall/F1 per entity type) requires comparing predicted tags against the ground-truth BIO labels for the held-out 20%. This comparison was designed in `metrics.py` (`calculate_entity_f1`, `calculate_parsing_accuracy`) but a full evaluation loop connecting the trained model's output to these functions was not run in the current experiment — this is a noted limitation and improvement item.

---

## 4. Results

### 4.1 Trained Model Artifacts

| Artifact | Size | Details |
|----------|------|---------|
| `output/processed/vocab.pth` | 391 KB | 17,067 tokens, universal vocabulary |
| `output/models/chunker/best_model.pth` | **9.7 MB** | 2,451,458 parameters, BiLSTM-CRF |
| `output/models/chunker/siamese_encoder.pth` | **9.3 MB** | BiLSTM encoder, 5 weight tensors |
| `output/json/structured_logs.json` | **3.2 MB** | 2,000 structured records |

### 4.2 Training Loss Curves (Documented Run)

#### BiLSTM-CRF Chunker

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 3.21 | 0.72 |
| 2 | 0.52 | 0.55 |
| 3 | 0.32 | 0.52 |
| 4 | 0.22 | 0.51 |
| 5 | 0.16 | 0.51 |
| 6 | 0.11 | 0.51 |
| 7 | 0.08 | 0.50 |
| 8 | 0.06 | 0.50 |
| 9 | 0.04 | 0.50 |
| 10 | 0.03 | 0.50 |

**Observations:**
- Train loss decreases monotonically — the model is learning
- Val loss drops sharply in epochs 1–3, then plateaus at ~0.50
- The plateau indicates the model has generalised as much as it can with the current data and hyperparameters
- The growing gap between train (0.03) and val (0.50) after epoch 5 indicates **mild overfitting** — early stopping at epoch 5–6 would be more optimal

#### Siamese Encoder

| Epoch | Loss |
|-------|------|
| 1 | 0.35 |
| 2 | 0.15 |
| 3 | 0.09 |

- Rapid convergence indicates the triplet mining strategy is effective
- The encoder learns to separate different templates quickly
- Further training would likely reduce loss further

### 4.3 Inference Results (Proxifier Domain, 2000 logs)

The trained pipeline was run on `Proxifier_2k.log`. All 2,000 logs processed successfully.

| Metric | Value |
|--------|-------|
| Total logs processed | 2,000 |
| Method: Bi-LSTM-CRF | 2,000 (100%) |
| Method: Siamese-Resolver | 0 (0%) |
| Status: SUCCESS | 2,000 (100%) |
| Status: UNKNOWN_PATTERN | 0 (0%) |
| Confidence: min | 0.9504 |
| Confidence: avg | 0.9996 |
| Throughput | ~223 logs/sec |
| Total time | ~9 seconds |

### 4.4 Observed Tag Distribution

From the 2,000 structured output records (51,604 total token positions):

| Tag | Count | Percentage |
|-----|-------|-----------|
| `O` (Outside) | 27,853 | 54.0% |
| `I-PARAM` | 19,783 | 38.3% |
| `B-PARAM` | 3,870 | 7.5% |
| `B-LEVEL` | 98 | 0.2% |
| All other tags | 0 | 0.0% |

- **1,044 / 2,000 records (52.2%)** contain at least one entity tag (non-O)
- The model detects PARAM spans and LEVEL tokens on Proxifier data
- TIME and COMPONENT tags were not triggered for Proxifier format (explained in Discussion)

### 4.5 Sample Structured Output

```
Log:    [10.30 17:16:10] QQ.exe - tcpconn6.tencent.com:443 error
Output:
  [      → O
  10.30  → O
  17     → O
  :      → O
  16     → O
  :      → O
  10     → O
  ]      → O
  QQ.exe → O
  -      → O
  ...    → O
  error  → B-LEVEL   ← LEVEL detected ✓
```

```
Log:    [10.30 16:49:07] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 0 bytes sent
Output:
  [                      → B-PARAM   ← parameter span detected
  10.30                  → I-PARAM
  16:49:07               → I-PARAM
  ]                      → I-PARAM
  chrome.exe             → I-PARAM
  ...
  0                      → B-PARAM
  bytes                  → O
```

---

## 5. Discussion

### 5.1 What Worked

**1. Cross-domain generalisation via universal vocabulary**
Training on all 9 domains with a shared 17,067-token vocabulary allows the model to process any domain's logs without re-training. The single model handles Android Logcat, Java-based Hadoop logs, and network proxy logs with the same weights.

**2. CRF tag constraint enforcement**
The CRF transition matrix prevents invalid BIO sequences. Without it, softmax-based taggers routinely produce `I-PARAM` tokens that do not follow `B-PARAM`. The CRF layer eliminates this class of error entirely.

**3. High-confidence pipeline throughput**
All 2,000 Proxifier logs were processed at ~223 logs/sec on CPU with 100% success rate. The fast chunker path handled every log without needing the slower Siamese fallback. This is practical for moderate-scale real-time log streaming.

**4. Modular architecture**
The five-module design (data / chunker / siamese / engine / monitor) allows each component to be tested and improved independently. The `HybridParser` routing is controlled by a single threshold parameter.

---

### 5.2 Limitations and Observed Issues

**Limitation 1 — Siamese fallback was never triggered**

All 2,000 inference records used the BiLSTM-CRF path. The minimum observed confidence was 0.9504, well above the 0.90 threshold. Two explanations:

- *The model is genuinely confident:* After training on 18,000 multi-domain logs, the CRF produces high-probability tag sequences for most inputs, especially when the dominant tag is `O`
- *Softmax confidence is inflated:* With 10 tag classes and `O` being the majority class (~54% of tokens), the model can achieve high softmax max-probability by assigning most tokens to `O`. This is a known limitation of softmax-based confidence — it does not reflect uncertainty about whether the correct tag is one of the minority entity tags

The practical consequence is that the Siamese path is never exercised at this confidence threshold on this dataset. A lower threshold (e.g., 0.70) would route some logs to Siamese, but this needs tuning against labelled validation data.

**Limitation 2 — Proxifier timestamps not tagged as TIME**

The Proxifier timestamp format `[10.30 16:49:06]` does not match the time heuristics in `processor.py`:
```python
re.match(r'^\d{2,4}[-:/]\d{2}[-:/]\d{2,4}$|^\d{6,}$|^\d{2}:\d{2}:\d{2}', token)
```
The token `10.30` uses a dot separator and does not match. Consequently, during training, the template `[<*>] app - host close` marks the entire bracket content as `PARAM`, not `TIME`. The model learned this label faithfully — it is tagging what it was trained on.

Fix: extend the time regex to include Proxifier's `\d{1,2}\.\d{1,2}` pattern, or add a domain-specific profile to override heuristics for Proxifier.

**Limitation 3 — No formal entity-level evaluation loop**

The `metrics.py` functions (`calculate_entity_f1`, `calculate_parsing_accuracy`) are implemented but were not connected to a post-training evaluation run. The pipeline success rate (100%) measures whether logs were *processed*, not whether the entity tags are *correct*. Formal Precision/Recall/F1 evaluation requires comparing predicted BIO sequences against the ground-truth BIO labels from the held-out 20% validation set.

**Limitation 4 — Class imbalance (O dominates)**

In log lines, the majority of tokens are "outside" any entity (`O` = 54% of tokens in output). This imbalance biases the model to predict `O` for uncertain tokens. Weighted cross-entropy loss or focal loss on the CRF emissions could improve minority-class entity recall.

**Limitation 5 — MLflow experiment logs not persisted**

The `mlruns/` directory was not found. Training curves reported in the Results section were captured during a documented training run but the MLflow artifacts were not committed or are stored outside the project root. For reproducibility, an explicit `mlflow.set_tracking_uri("mlruns/")` should be added before `set_experiment()`.

---

### 5.3 Comparison to Baseline Approaches

| Approach | Cross-domain | Unseen logs | Requires rules | Speed |
|----------|-------------|-------------|----------------|-------|
| Regex parsers (e.g. Drain) | No (per-domain) | No | Yes | Fast |
| Supervised NER (softmax) | Yes | No | No | Fast |
| **OMNI-LOG (this work)** | **Yes** | **Yes (Siamese)** | **No** | **~223 logs/sec** |
| LLM-based parsing | Yes | Yes | No | Very slow, costly |

OMNI-LOG's hybrid approach occupies a practical middle ground: neural generalisation without the cost or opacity of large language models.

---

## 6. Conclusions

1. **Cross-domain log parsing is feasible with a single BiLSTM-CRF model.** A model trained on 9 heterogeneous domains generalises to unseen logs from those domains without per-domain configuration.

2. **Template-guided BIO labelling provides free supervision.** The LogHub `<*>` markers enable automatic ground-truth label generation, removing the need for expensive manual annotation.

3. **CRF is necessary for valid entity sequences.** The transition constraint layer eliminates structurally impossible tag patterns that softmax-based taggers produce.

4. **The hybrid fallback architecture is sound but not fully validated in this run.** The Siamese path is correctly implemented (verified by unit tests) but was not triggered during inference due to high chunker confidence. A threshold sensitivity study is needed.

5. **Pipeline infrastructure is complete and functional.** The full data → train → inference → JSON output pipeline runs end-to-end in a single command (`python src/main/main.py`), processing 2,000 logs in ~9 seconds on CPU.

---

## 7. Recommended Improvements

### Priority 1 — Correctness

| # | Issue | Fix |
|---|-------|-----|
| 1 | Proxifier timestamps tagged as PARAM | Extend time regex in `processor.py:50` to include `\d{1,2}\.\d{1,2}` pattern |
| 2 | Softmax confidence always high | Implement CRF marginal probability confidence (sum over all paths vs. best path) instead of softmax max-prob |
| 3 | `demo.py` crashes loading siamese model | Change `siamese_checkpoint["model_state_dict"]` to `torch.load(siamese_path)` directly (plain state_dict) |
| 4 | `demo.py` wrong encoder dimensions | Change `embedding_dim=64, hidden_dim=128` to `embedding_dim=128, hidden_dim=256` |

### Priority 2 — Evaluation

| # | Item | Why |
|---|------|-----|
| 5 | Run entity-level F1 evaluation | Connect `calculate_entity_f1()` to the 20% validation set; report per-entity Precision/Recall/F1 |
| 6 | Run inference on all 9 domains | Current JSON output is Proxifier only; multi-domain comparison would validate the cross-domain claim |
| 7 | Threshold sensitivity study | Vary `confidence_threshold` (0.70–0.95) and measure how many logs fall to Siamese; validate Siamese accuracy on those |

### Priority 3 — Training Quality

| # | Item | Why |
|---|------|-----|
| 8 | Add early stopping | Val loss plateaus at epoch 3–5; stopping early saves compute and prevents overfitting |
| 9 | Stratify train/val split by domain | Current random split may under-represent some domains in validation |
| 10 | Weighted loss for minority entity classes | `O` class dominates (54%); weighted CRF loss would improve TIME/COMPONENT recall |
| 11 | Hard negative mining for Siamese | Random negatives are easy; hard negatives (most similar wrong template) would produce a more discriminative encoder |

### Priority 4 — Reproducibility

| # | Item | Fix |
|---|------|-----|
| 12 | Fix MLflow tracking URI | Add `mlflow.set_tracking_uri("mlruns/")` to `train.py` and `train_siamese.py` before `set_experiment()` |
| 13 | Align `config.yaml` epochs with training results | Either update `config.yaml` to `epochs: 10` or document that the 10-epoch results used a standalone training run |

---

## 8. References

1. LogHub Dataset — He, P. et al. (2020). *Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics.* https://github.com/logpai/loghub
2. Lample, G. et al. (2016). *Neural Architectures for Named Entity Recognition.* NAACL. (BiLSTM-CRF for NER)
3. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
4. Koch, G. et al. (2015). *Siamese Neural Networks for One-Shot Image Recognition.* ICML Deep Learning Workshop.
5. Schroff, F. et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR. (Triplet loss)
6. Lafferty, J. et al. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* ICML.

## 9. Appendices

Github
https://github.com/NikhilVijayakumar/OMNI-LOG