# OMNI-LOG Training Results

> **See also:**
> - [Preprocessing Pipeline](preprocessing.md) — tokenization, BIO tag generation, vocab building
> - [Theoretical Background](theory.md) — BiLSTM, CRF, Siamese, Triplet Loss, BIO tagging, references

---

## Pipeline Artifact Flow

| Phase | Produces |
|-------|----------|
| Preprocessing (`data`) | `output/processed/vocab.pth` — universal token vocabulary (17,067 entries) |
| Training (`train`) | `output/models/chunker/best_model.pth` — BiLSTM-CRF weights + vocab + tag map |
| Training (`train`) | `output/models/chunker/siamese_encoder.pth` — Siamese encoder weights |
| Inference (`inference`) | `output/json/<domain>.jsonl` — structured parsed log records |

---

## Chunker (BiLSTM-CRF)

**Architecture:** `BiLSTM-CRF` — Embedding (128-dim) → Bidirectional LSTM (256-dim hidden) → Linear projection → CRF (Viterbi decode at inference)

**Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `embedding_dim` | 128 | Token embedding size |
| `hidden_dim` | 256 | 128 per direction (bidirectional) |
| `learning_rate` | 0.001 | Adam optimizer |
| `epochs` | 10 | Full passes over training set |
| `batch_size` | 32 | Logs per gradient step |
| `optimizer` | Adam | Gradient clip max_norm=5.0 |
| `train/val split` | 80/20 | `random_split` with seed=42 |
| `dropout` | 0.1 | Applied after LSTM output |
| `num_layers` | 1 | Single BiLSTM layer |

**Loss Curves:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 3.21      | 0.72     |
| 2     | 0.52      | 0.55     |
| 3     | 0.32      | 0.52     |
| 4     | 0.22      | 0.51     |
| 5     | 0.16      | 0.51     |
| 6     | 0.11      | 0.51     |
| 7     | 0.08      | 0.50     |
| 8     | 0.06      | 0.50     |
| 9     | 0.04      | 0.50     |
| 10    | 0.03      | 0.50     |

**Final:** Train loss=0.0326, Val loss=0.5026
**Training time:** ~2-3 min on CPU for 10 epochs

**Observations:**
- Train loss drops steadily, model fits training data well
- Val loss plateaus around 0.50 after epoch 3, indicating the model learns the CRF transition constraints early
- Gap between train and val loss suggests slight overfitting after epoch 5; early stopping around epoch 5-6 would be optimal

## Siamese Encoder

**Architecture:** Shared `LogEncoder` (Embedding → BiLSTM → Masked Mean Pool → L2 Normalize) used in a Siamese configuration with Triplet Loss training.

**Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `embedding_dim` | 64 | Shared embedding size |
| `hidden_dim` | 128 | 64 per direction |
| `learning_rate` | 1e-4 | Adam optimizer |
| `epochs` | 3 | Triplet training passes |
| `loss` | TripletMarginLoss | margin=1.0, p=2 (Euclidean) |
| `negative mining` | Random | Any non-matching template sampled per anchor |

**Loss Curve:**

| Epoch | Loss    |
|-------|---------|
| 1     | 0.35    |
| 2     | 0.15    |
| 3     | 0.09    |

**Final:** Loss=0.0888
**Training time:** ~1-2 min on CPU for 3 epochs

**Observations:**
- Loss drops rapidly, indicating the encoder learns to separate different templates quickly
- Triplet mining with random negative sampling is effective
- Training for more epochs would likely improve further

---

## Training Infrastructure

- **Experiment tracking:** MLflow — runs logged under `OMNI-LOG-Chunker` and `OMNI-LOG-Siamese`
- **Artifacts logged:** model checkpoints, hyperparameters, per-epoch metrics
- **Device:** CPU (CUDA used automatically if available)
- **Reproducibility seeds:** `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`
- **MLflow run data:** stored locally in `mlruns/` directory

---

## Inference Performance

**Test Domain:** Proxifier (2000 logs)
**Device:** CPU
- Total logs: 2000
- Throughput: 223.7 logs/sec
- Total time: ~9 seconds
- Success rate: 100%
- Average confidence: 99.96%
- Model: `output/models/chunker/best_model.pth` (vocab=17067)
