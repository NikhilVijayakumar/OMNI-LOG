# OMNI-LOG Training Results

## Chunker (BiLSTM-CRF)

**Hyperparameters:**
- embedding_dim: 128
- hidden_dim: 256
- learning_rate: 0.001
- epochs: 10
- batch_size: 32
- optimizer: Adam (gradient clip 5.0)
- train/val split: 80/20 (seed=42)

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

**Hyperparameters:**
- embedding_dim: 64
- hidden_dim: 128
- learning_rate: 0.001
- epochs: 3
- loss: TripletMarginLoss (margin=1.0)

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

## Inference Performance

**Test Domain:** Proxifier (2000 logs)
**Device:** CPU
- Total logs: 2000
- Throughput: 223.7 logs/sec
- Total time: ~9 seconds
- Success rate: 100%
- Average confidence: 99.96%
- Model: `output/models/chunker/best_model.pth` (vocab=17067)
