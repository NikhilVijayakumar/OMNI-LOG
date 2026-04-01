# OMNI‑LOG  
**Multi‑Category Log Structuring Pipeline (MLSP)**  
*A Cross‑Domain NLP Engine using Bi‑LSTM Chunking and Siamese Template Resolution*

***

## 📌 Overview

**OMNI‑LOG** is an NLP‑based pipeline that automatically structures system logs from **six domains** (HDFS, Android, BGL, Spark, Windows, and Apache) into standardized JSON. Instead of regex‑based parsing, it uses:

- A **multi‑task Bi‑LSTM chunker** (Bi‑LSTM‑CRF) to extract semantic entities across all domains.  
- A **Siamese vector resolver** that matches “unseen” log variants to templates in a multi‑domain library.  
- A **configurable batching engine** that tunes `batch_size` to optimize throughput and latency for different log densities (e.g., Spark vs Windows).  

This design demonstrates that system logs share a **common underlying grammar** and can be processed with a unified neural engine.

***

## 🚀 Core components

### 1. Multi‑Task Bi‑LSTM Chunker (The “Extractor”)

A **single Bi‑LSTM‑based sequence‑labeling model** trained on all six log domains as a **multi‑task NER‑style problem**.

- **Task**: For each log line, predict BIO‑tags over shared semantic entities:
  - `TIME` (timestamps)
  - `LEVEL` (log levels: `INFO`, `ERROR`, `WARN`, etc.)
  - `COMPONENT` (module or subsystem, e.g., `SparkDriver`, `HDFSNameNode`)
  - `PARAM` (variable‑like tokens: IDs, paths, numbers)
- **Label scheme**:
  - `B‑TIME`, `I‑TIME`, `B‑LEVEL`, `I‑LEVEL`, `B‑COMPONENT`, `I‑COMPONENT`, `B‑PARAM`, `I‑PARAM`, `O` (other).  
  - Labels are **shared across all categories** to enforce cross‑domain learning.
- **Architecture**:
  - Word embedding → Bi‑LSTM (bidirectional) → optional **CRF layer** over tags (recommended). [domino](https://domino.ai/blog/named-entity-recognition-ner-challenges-and-model)
  - Optional **character‑level CNN** for noisy tokens (hex IDs, random strings). [emergentmind](https://www.emergentmind.com/topics/bilstm-crf-model)
- **Multi‑task strategy**:
  - All six datasets are mixed in training batches, optionally tagged with domain ID (`HDFS`, `Spark`, etc.).  
  - This lets the model learn **domain‑invariant features** while still handling domain‑specific phrases.

***

### 2. Siamese Vector Resolver (The “Robustifier”)

A **Siamese‑style similarity model** that encodes log lines and templates into vectors and retrieves the best‑matching template.

- **Encoder**:
  - Shared Bi‑LSTM encoder (same or derived from the chunker) with **mean‑ or last‑state pooling** into a fixed‑size vector. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1155/2022/7056149)
- **Template library**:
  - Built from **ground‑truth templates** in the LogHub datasets (e.g., `HDFS_templates.csv`, `Spark_templates.csv`). [github](https://github.com/logpai/loghub/blob/master/HDFS/README.md)
  - Each entry: `(template_id, template_text, template_vector, domain)`.  
- **Training**:
  - Uses **contrastive loss** or **triplet loss** over `(log, template, other_template)` triplets. [arxiv](https://arxiv.org/pdf/1901.08109.pdf)
- **Inference**:
  - For a new log line:
    1. Encode it into a vector.  
    2. Compute **cosine similarity** with all template vectors.  
    3. If similarity ≥ threshold → assign the corresponding template.  
    4. Else fall back to pure Bi‑LSTM extractions or mark as “new template candidate”.

This enables OMNI‑LOG to handle **unseen log variants** and **cross‑domain template drift**.

***

### 3. Hyperparameter‑Driven Batching Engine (The “Optimizer”)

A configurable engine that controls **throughput** and **latency** via `batch_size` and JSON‑batching parameters.

- **Batching semantics**:
  - **Input batching**: Reads `batch_size` lines from a file (or domain‑filtered stream).  
  - **Model batching**: Packs tokens into tensors for the Bi‑LSTM.  
  - **Output batching**: Writes structured JSON outputs in batches (e.g., 500 lines / JSON array file).  
- **Configurable parameters**:
  - `batch_size`: Inference batch size (e.g., 512 for Spark, 16 for Windows).  
  - `write_batch_size`: Number of JSON objects per file.  
  - `max_seq_len`: Maximum token length after preprocessing.  
- **Metrics**:
  - Lines per second (`lines/sec`), latency per batch (`ms`), memory usage vs `batch_size`.  

This aligns with your “configurable JSON batching” requirement and supports **throughput/latency analysis** in your thesis.

***

## 📂 Data sources and preprocessing (LogHub)

- **Dataset**: [LogHub – A Large Collection of System Log Datasets for AI‑Driven Log Analysis](https://github.com/logpai/loghub) [zenodo](https://zenodo.org/records/8196385)
- **Domains**:  
  - HDFS  
  - Android  
  - BGL  
  - Spark  
  - Windows  
  - Apache  
- **Preprocessing pipeline**:
  1. Download raw logs and template files (e.g., `HDFS_2k.log`, `HDFS_templates.csv`). [github](https://github.com/logpai/loghub/blob/master/HDFS/README.md)
  2. Tokenize lines (split on whitespace, keep meaningful special characters).  
  3. Build a **shared vocabulary** over all six datasets (with `UNK` token). [emergentmind](https://www.emergentmind.com/topics/bilstm-crf-model)
  4. Generate **BIO labels**:
     - Heuristics or template‑alignment maps each line to `TIME`, `LEVEL`, `COMPONENT`, `PARAM`.  
     - Store as `(tokens, bio_tags, domain_label)`.  
  5. Optional character‑level inputs for unknown tokens.  
- **Splits**:
  - Use existing LogHub splits where available; otherwise, 60/20/20 per dataset. [deepwiki](https://deepwiki.com/logpai/loghub/1.2-accessing-the-datasets)

***

## 🧠 Model architecture (PyTorch)

### 1. Bi‑LSTM Chunker (Bi‑LSTM‑CRF)

```python
import torch
import torch.nn as nn
from torchcrf import CRF  # optional but recommended

class BiLSTMChunker(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        embedding_dim=128,
        hidden_dim=256,
        use_crf=True,
        char_vocab_size=None,
        char_embed_dim=32,
        use_char_cnn=False,
    ):
        # ... (same as previous Bi‑LSTM sketch with CRF & char‑CNN)
    def forward(self, words, chars=None, lengths=None):
        # ... (same as previous)
```

**Key points**:
- CRF is recommended for valid tag‑sequence constraints. [aclanthology](https://aclanthology.org/Y18-1061.pdf)
- Char‑level CNN helps with noisy log tokens. [emergentmind](https://www.emergentmind.com/topics/bilstm-crf-model)

***

### 2. Siamese Resolver

```python
import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        # ... (same as previous Bi‑LSTMEncoder)

class SiameseResolver(nn.Module):
    def __init__(self, encoder):
        # ... (same as previous Siamese class)
    def forward(self, log_a, log_b, len_a, len_b):
        # ... (same as previous)
```

**Training**:
- Use **triplet** or **contrastive loss**; templates are stored in a vector DB (or simple matrix). [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1155/2022/7056149)

***

## 📊 Evaluation, metrics, and MLOps

### Key metrics

- **Per‑category Parsing Accuracy (PA)**:
  - Percentage of logs where all entities match the true template.  
- **Entity‑level F1**:
  - F1 for `TIME`, `LEVEL`, `COMPONENT`, `PARAM`.  
- **Template matching metrics**:
  - Template recall / precision for the Siamese resolver.  
- **Throughput & latency**:
  - `lines/sec`, latency per batch vs `batch_size`.  

### Tools

- **MLflow**: Track sweeps over `batch_size`, `hidden_dim`, `embedding_dim`, CRF‑on/off.  
- **DVC**: Version‑control the six LogHub datasets and preprocessed files. [zenodo](https://zenodo.org/records/8196385)
- **Logging**: Structured JSON outputs can be reused for downstream analysis.  

***

## 🛠 Design principles

- **No LLM / No Agents**:
  - OMNI‑LOG is built purely with **Bi‑LSTM‑CRF**, **Siamese networks**, and **multi‑domain template libraries**.  
  - This ensures **reproducibility**, **explainability**, and thesis‑grade clarity.  
- **Cross‑domain generalization**:
  - Single model trained on all six domains proves **shared log grammar**.  
- **Configurable deployment**:
  - You can tune `batch_size` per domain (e.g., large for Spark, small for Windows) to balance latency and throughput.  

***

## 🛠️ Module-by-Module Breakdown

### 1. `data` — The Foundation
This is where you handle the "dirty work." By isolating this, you ensure that your models receive clean, standardized inputs regardless of whether the log is from an Android device or an Apache server.
* **Key Task:** Converting LogHub's raw text and CSV templates into **BIO-tagged sequences** ($B-TIME, I-TIME, O$).
* **Innovation Proof:** This module creates the **Shared Vocabulary** that proves logs have a "common underlying grammar."

### 2. `chunker` — The "Brain" (Bi-LSTM-CRF)
This is your primary neural innovation. Keeping it separate allows you to experiment with different architectures (like adding a Character-CNN) without touching the data loading or batching logic.
* **Technical Edge:** By using a **CRF (Conditional Random Field)** layer here, you move beyond simple classification to "sequence optimization," which is a high-tier NLP technique.


### 3. `siamese` — The "Safety Net" (Similarity)
This module implements the **Metric Learning** aspect of your proposal. It takes the embeddings from the chunker and maps them into a vector space.
* **Key Task:** Building the **Template Library**. You store the "ideal" vectors for known log types here.
* **Innovation Proof:** This is your solution for "unseen" logs, proving the system is robust and not just a memorization engine.


### 4. `engine` — The "Heart" (Batching & JSON)
This is where your **Hyperparameter-Driven Batching** lives. This module is the "conductor" that coordinates the chunker and the resolver.
* **Key Task:** Implementing the logic: *If Chunker Confidence < Threshold, then call Siamese Resolver.*
* **JSON Serialization:** It gathers the predicted entities and wraps them into the structured JSON format you defined.

### 5. `monitor` — The "Proof" (MLOps)
In an M.Tech project, the results are as important as the code. This module tracks your experiments.
* **Key Task:** Using **MLflow** to plot the "Batch Size vs. Throughput" graph.
* **Innovation Proof:** It provides the empirical data to show your professor exactly how your hyperparameters affect system performance.

---

## 📂 Recommended Directory Structure
To keep your implementation clean, I suggest the following folder layout:

```text
omni-log/
├── data/
│   ├── raw/                # Original LogHub files
│   ├── processed/          # BIO-tagged tensors
│   └── loader.py           # Dataset & DataLoader classes
├── chunker/
│   ├── model.py            # Bi-LSTM-CRF class
│   └── train.py            # Chunker training script
├── siamese/
│   ├── encoder.py          # Vector embedding logic
│   └── resolver.py         # Similarity & Template library
├── engine/
│   ├── batch_config.py     # Hyperparameter settings
│   └── pipeline.py         # The main 'process_logs' logic
└── monitor/
    ├── metrics.py          # PA and F1-Score calculations
    └── mlflow_utils.py     # Experiment tracking
```


## 🖥 How to run (CLI outline)

```bash
# 1. Install dependencies
pip install torch torchcrf scikit-learn mlflow dvc pandas

# 2. Download LogHub data
python scripts/download_loghub.py \
    --categories HDFS Android BGL Spark Windows Apache

# 3. Preprocess and label
python scripts/preprocess.py \
    --data_dir ./data/loghub \
    --output_dir ./data/omni-log

# 4. Train Bi‑LSTM chunker
python train_bilstm.py \
    --data ./data/omni-log \
    --model_dir ./models/bilstm \
    --use_crf True \
    --domain_labels True

# 5. Build Siamese template library
python build_template_library.py \
    --templates ./data/loghub/templates \
    --model_dir ./models/bilstm \
    --library_file ./models/template_library.pt

# 6. Run inference engine (with batching)
python engine.py \
    --input ./data/raw/Spark.spark \
    --model_dir ./models/bilstm \
    --template_library ./models/template_library.pt \
    --batch_size 512 \
    --write_batch_size 500 \
    --output_format json
```

***


