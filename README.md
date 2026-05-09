# OMNI-LOG

### Multi-Category Log Structuring Pipeline (MLSP)

*A Cross-Domain NLP Engine using Bi-LSTM-CRF and Siamese Template Resolution*

---

## 📌 Overview

**OMNI-LOG** is a unified NLP pipeline designed to automatically structure heterogeneous system logs into standardized JSON format.

Unlike traditional regex-based parsers, OMNI-LOG leverages **neural sequence modeling and metric learning** to generalize across multiple log domains.

### 🔑 Core Capabilities

* **Cross-domain log parsing** using a single model
* **Semantic entity extraction** via Bi-LSTM-CRF
* **Robust handling of unseen logs** using Siamese similarity matching
* **Configurable batching engine** for throughput vs latency optimization

---

## 🧠 System Architecture

OMNI-LOG is built on three core components:

### 1. Bi-LSTM-CRF Chunker (Extractor)

* Performs **sequence labeling (BIO tagging)**
* Identifies:

  * `TIME`
  * `LEVEL`
  * `COMPONENT`
  * `PARAM`

---

### 2. Siamese Vector Resolver (Robustifier)

* Encodes logs into vector space
* Matches logs to **template library** using cosine similarity
* Handles **unseen or ambiguous log patterns**

---

### 3. Batching & Execution Engine (Optimizer)

* Controls:

  * `batch_size`
  * `write_batch_size`
* Optimizes:

  * throughput (`logs/sec`)
  * latency (`ms/batch`)

---

## 📂 Project Structure

```text
omni-log/
│
├── data/
│   ├── logs/                # Raw log files (LogHub 2k subsets)
│   ├── docs/                # Module-level documentation
│
├── src/
│   ├── features/
│   │   ├── chunker/         # Bi-LSTM-CRF model
│   │   │   ├── model.py
│   │   │   └── train.py
│   │   │
│   │   ├── data/            # Data loading & preprocessing
│   │   │   └── loader.py
│   │   │
│   │   ├── engine/          # Pipeline & batching logic
│   │   │   ├── pipeline.py
│   │   │   └── batch_config.py
│   │   │
│   │   ├── monitor/         # Metrics & MLflow tracking
│   │   │   ├── metrics.py
│   │   │   └── mlflow_utils.py
│   │   │
│   │   ├── siamese/         # Similarity-based resolver
│   │   │   ├── encoder.py
│   │   │   └── resolver.py
│   │
│   └── __init__.py
```

---

## 📊 Dataset

* **Source:** LogHub dataset
* **Subset Used:** `*_2k.log` files for controlled experimentation

### Domains Included

* Android
* Apache
* Hadoop
* HealthApp
* HPC
* Linux
* OpenSSH
* Proxifier
* Zookeeper

---

### 🎯 Dataset Strategy

To ensure **balanced multi-domain learning**, this project uses standardized 2k log subsets:

* Prevents dominance of large datasets
* Enables fair cross-domain training
* Reduces computational overhead
* Supports rapid experimentation and reproducibility

---

## 🔄 Pipeline Workflow

```text
Raw Logs
   ↓
Tokenization & Normalization
   ↓
BIO Tag Generation
   ↓
Bi-LSTM-CRF Chunker
   ↓
[Confidence Check]
   ├── High → Direct Output
   └── Low  → Siamese Resolver
                    ↓
             Template Matching
                    ↓
Structured JSON Output
```

---

## ⚙️ Key Features

### 🔹 Cross-Domain Generalization

A single model learns shared log grammar across multiple systems.

### 🔹 Robustness to Unseen Logs

Siamese network enables template matching for unknown patterns.

### 🔹 Configurable Performance

Batching parameters allow tuning for:

* real-time systems (low latency)
* batch processing (high throughput)

---

## 📊 Evaluation Metrics

* **Parsing Accuracy (PA)**
* **Entity-level F1 Score**
* **Template Matching Accuracy**
* **Throughput (logs/sec)**
* **Latency (ms/batch)**

---

## 🛠️ Installation

```bash
pip install -e .
```

> **Note for new contributors:** The `output/` directory (models, vocab, JSON results) is **gitignored** and will not exist after a fresh clone. You must run the training pipeline first to generate all artifacts before running inference.

---

## ▶️ Quick Demo (Inference on a Log File)

Use `demo.py` to run the trained pipeline against any log file. It automatically loads the trained models from `output/` and writes structured JSON.

```bash
# Run on any specific log file
python demo.py --input data/logs/Hadoop_2k.log

# Run on a different domain with a custom output path
python demo.py --input data/logs/Linux_2k.log --output output/json/linux_structured.json

# Run on the first log file found automatically
python demo.py
```

If trained models are not found in `output/`, `demo.py` will trigger the full training pipeline automatically.

**Output format** (`output/json/<domain>_structured.json`):

```json
{
    "raw": "2015-10-18 18:01:48 INFO NameSystem: completeFile: blk_-160",
    "structured": {
        "time": "2015-10-18",
        "level": "INFO",
        "component": null,
        "params": ["blk_-160"]
    },
    "metadata": {
        "method": "Bi-LSTM-CRF",
        "confidence": 0.996,
        "status": "SUCCESS"
    }
}
```

---

## ⚙️ Full Training Pipeline

To train from scratch (data → train → inference):

**Step 1:** Set the phase in `src/main/config/config.yaml`:

```yaml
pipeline:
  phase: "all"   # options: "data" | "train" | "inference" | "all"
```

**Step 2:** Run from the project root:

```bash
python src/main/main.py
```

### Generated Artifacts

After a successful run the following files are created (all gitignored):

```
output/
├── processed/
│   └── vocab.pth                        ← Universal vocabulary (17,067 tokens)
├── models/
│   └── chunker/
│       ├── best_model.pth               ← Trained BiLSTM-CRF (~9.7 MB, 2.4M params)
│       └── siamese_encoder.pth          ← Trained Siamese encoder (~9.3 MB)
└── json/
    └── structured_logs.json             ← Structured output from inference phase
```

### Running Individual Phases

```yaml
phase: "data"       # Build vocab only → output/processed/vocab.pth
phase: "train"      # Train models only (requires vocab) → output/models/
phase: "inference"  # Run inference only (requires trained models) → output/json/
phase: "all"        # Full pipeline end-to-end
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

Individual test files:

```bash
pytest tests/unit/siamese/test_hybrid.py -v    # Hybrid routing logic
pytest tests/unit/chunker/test_model.py -v     # BiLSTM-CRF model
pytest tests/integration/test_end_to_end.py -v # Full pipeline
```

---

## 📈 Experiment Tracking

* **MLflow** → hyperparameter tuning & performance tracking (runs saved to `mlruns/`)
* **DVC** → dataset versioning

---

## 🧠 Design Principles

* **No LLMs / No Agents**
* Fully reproducible classical deep learning pipeline
* Focus on **interpretability and control**

---

## 🏁 Key Contribution

> OMNI-LOG demonstrates that heterogeneous system logs can be modeled using a **shared semantic grammar**, enabling a single neural architecture to generalize across domains while maintaining robustness to unseen log patterns.

---

## 📎 References

* LogHub Dataset: https://github.com/logpai/loghub
* BiLSTM-CRF for Sequence Labeling
* Siamese Networks for Metric Learning

---


 python demo.py --input data/logs/HealthApp_2k.log --output output/json/healthApp_structured.json

