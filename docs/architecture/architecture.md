# OMNI-LOG Architecture

## System Overview

OMNI-LOG is a cross-domain NLP pipeline for structuring system logs into standardized JSON. It uses a hybrid architecture combining a BiLSTM-CRF sequence labeler (chunker) with a Siamese neural network for template-based resolution.

## Architecture Diagram

```
                     Raw Log Files (*.log) + Templates (*_templates.csv)
                                   |
                                   v
                     +-----------------------------+
                     |      DATA MODULE            |
                     |  - Regex Tokenization       |
                     |  - Template Alignment       |
                     |  - BIO Tag Generation       |
                     |  - Universal Vocabulary     |
                     |  - Tensor Conversion        |
                     +-----------------------------+
                                   |
                                   v
                     +-----------------------------+
                     |     CHUNKER MODULE          |
                     |  - BiLSTM-CRF Model         |
                     |  - Sequence Labeling        |
                     |  - Entity Extraction        |
                     |  - Confidence Scoring       |
                     +-----------------------------+
                                   |
                          +--------+--------+
                          |                 |
                     Confidence        Confidence
                     >= 0.90           < 0.90
                          |                 |
                          v                 v
                     +--------+    +------------------+
                     | Direct |    | SIAMESE MODULE   |
                     | Output |    | - LogEncoder     |
                     +--------+    | - Template Lib   |
                                   | - Cosine Sim     |
                                   +------------------+
                          |                 |
                          +--------+--------+
                                   |
                                   v
                     +-----------------------------+
                     |      ENGINE MODULE          |
                     |  - Batch Streaming          |
                     |  - HybridParser Routing     |
                     |  - JSONWriter Output        |
                     |  - Performance Stats        |
                     +-----------------------------+
                                   |
                                   v
                        Structured JSON Output
```

## Module Interface

### 1. Data Module (`src/features/data/`)

**Components:**
- `processor.py` — `LogProcessor`: tokenization, BIO tagging, numericalization, vocabulary
- `loader.py` — `get_dataloader()`: multi-domain ConcatDataset + DataLoader
- `constants.py` — TAG_MAP, LOG_PROFILES, DOMAIN_TO_IDX

**Input:** Raw log files + template CSVs
**Output:** Batched tensors (tokens, tags, lengths, domain IDs)

### 2. Chunker Module (`src/features/chunker/`)

**Components:**
- `model.py` — `BiLSTM_CRF`: Embedding → BiLSTM → Linear → CRF → Viterbi decode
- `train.py` — Training loop with 80/20 split, MLflow logging

**Input:** Token ID sequences
**Output:** BIO tag sequences + confidence scores

### 3. Siamese Module (`src/features/siamese/`)

**Components:**
- `encoder.py` — `LogEncoder`: BiLSTM with masked mean pooling + L2 normalization
- `resolver.py` — `TemplateResolver`: template library, cosine similarity matching
- `hybrid_logic.py` — `HybridParser`: confidence-based routing orchestration

**Input:** Raw log lines
**Output:** Template ID, similarity score, or UNKNOWN_PATTERN

### 4. Engine Module (`src/features/engine/`)

**Components:**
- `pipeline.py` — `Pipeline`: process_file() streaming orchestration
- `batch_config.py` — `BatchConfig`: configuration dataclass with YAML loading
- `stream_handler.py` — `BatchStreamer`: line-by-line batch generator; `JSONWriter`: buffered JSON array writer

**Input:** Log file path, configuration
**Output:** `structured_logs.json` + processing stats

### 5. Monitor Module (`src/features/monitor/`)

**Components:**
- `metrics.py` — Parsing accuracy, Entity F1, Template accuracy, PerformanceMonitor
- `mlflow_utils.py` — `MLflowTracker`: experiment logging wrapper

**Input:** Predictions + ground truth
**Output:** Evaluation scores, MLflow experiment logs

## Data Flow

```
1. Data Ingestion
   Log File → BatchStreamer → line batches

2. Tokenization
   Raw line → LogProcessor.tokenize() → token list

3. Chunker Inference
   Token IDs → BiLSTM_CRF → BIO tags + confidence

4. Routing Decision
   confidence >= 0.90 → use chunker output
   confidence < 0.90 → fall back to Siamese resolver

5. Siamese Resolution
   Log line → LogEncoder → vector
   → dot product with template library → best match
   similarity >= 0.85 → use matched template
   otherwise → UNKNOWN_PATTERN

6. Output
   Structured record → JSONWriter → structured_logs.json
```

## Training Pipeline

```
1. Load data from all 9 domains
2. Build universal vocabulary across domains
3. Generate BIO tags via template alignment + heuristics
4. Split 80/20 train/validation (stratified by domain)
5. Train BiLSTM-CRF with CRF negative log-likelihood loss
6. Train LogEncoder with TripletMarginLoss
7. Log metrics to MLflow
8. Save model checkpoints
```

## Inference Pipeline

```
1. Load trained models + template library
2. Stream input log file in batches
3. For each log: tokenize → chunker → confidence check → route
4. Write structured JSON output in batches
5. Return throughput/success statistics
```

## Key Design Decisions

1. **Sequence Labeling over Classification:** Formulating log parsing as NER enables fine-grained entity extraction in a single pass

2. **CRF over Softmax:** CRF enforces valid tag transitions (e.g., I-TIME must follow B-TIME), eliminating impossible sequences

3. **Hybrid Architecture:** Chunker handles high-confidence cases fast; Siamese fallback catches edge cases without retraining

4. **Masked Mean Pooling:** Averaging only real tokens (not padding) produces length-invariant log representations

5. **L2 Normalization:** Makes cosine similarity a simple dot product, enabling efficient batched matrix multiplication during template lookup

6. **Configurable Batching:** Batch size controls throughput vs latency tradeoff; configurable via YAML

7. **Multi-Domain Training:** Single model trained on all 9 domains generalizes across heterogeneous log formats
