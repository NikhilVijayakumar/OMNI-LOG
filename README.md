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
pip install torch torchcrf scikit-learn pandas mlflow dvc
```

---

## ▶️ Usage (Pipeline Execution)

```bash
python src/features/engine/pipeline.py \
    --input data/logs/Hadoop_2k.log \
    --batch_size 256 \
    --write_batch_size 500 \
    --output output.json
```

---

## 🧪 Training (Chunker)

```bash
python src/features/chunker/train.py \
    --data_dir data/processed \
    --model_dir models/chunker \
    --use_crf True
```

---

## 📈 Experiment Tracking

* **MLflow** → hyperparameter tuning & performance tracking
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
