## 📂 Module 1: `data` — Unified Log Preprocessing Layer

---

## 📌 Purpose

The `data` module is responsible for transforming heterogeneous raw logs into a **structured, model-ready representation** using a unified semantic labeling scheme.

It serves as the **foundation of the OMNI-LOG pipeline**, ensuring that logs from multiple domains can be processed by a single neural architecture.

---

## 🎯 Objectives

* Normalize diverse log formats into a **shared representation**
* Generate **BIO-tagged sequences** for supervised learning
* Build a **universal vocabulary** across domains
* Prepare **batched tensor inputs** for the chunker model

---

## 📥 Input

* Raw log files (`*.log`) from LogHub (2k subsets)
* Corresponding template files (`*_templates.csv`)

### Supported Domains

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

## 📤 Output

Processed dataset stored as tensors:

```python
{
  "tokens": List[int],
  "tags": List[int],
  "length": int,
  "domain": int
}
```

Saved formats:

* `.pt` (PyTorch)
* `.npz` (NumPy)

---

## 🔄 Processing Pipeline

```text
Raw Logs + Templates
        ↓
Tokenization
        ↓
Template Alignment
        ↓
BIO Tag Generation
        ↓
Vocabulary Mapping
        ↓
Padding & Tensor Conversion
        ↓
Saved Dataset
```

---

## 1. Data Acquisition

The module uses datasets from LogHub.

To ensure **balanced and efficient experimentation**, only `*_2k.log` subsets are used.

### Rationale

* Prevents dataset imbalance
* Reduces computational overhead
* Enables reproducible experiments
* Ensures equal domain representation

---

## 2. Tokenization & Normalization

### Challenge

System logs contain:

* timestamps
* IP addresses
* file paths
* unique identifiers

Naive tokenization breaks semantic structure.

---

### Approach

A **regex-based tokenizer** is used to preserve meaningful tokens:

* `/var/log/syslog`
* `blk_-12345`
* `192.168.1.1`
* `ERROR:`

---

### Example

```text
Input Log:
INFO Received block blk_-123 from 192.168.0.1

Tokens:
["INFO", "Received", "block", "blk_-123", "from", "192.168.0.1"]
```

---

## 3. Universal Vocabulary Construction

A single vocabulary is built across all domains.

### Special Tokens

* `<PAD>` → sequence padding
* `<UNK>` → unknown tokens
* `<SOS>` / `<EOS>` → sequence boundaries

---

### Key Insight

Using a shared vocabulary enforces:

> Semantic consistency across domains (e.g., "INFO" has the same meaning everywhere)

This is essential for **cross-domain generalization**.

---

## 4. Template Alignment

Each raw log is aligned with its corresponding template.

### Example

```text
Log:      Receiving block blk_-160
Template: Receiving block <*>
```

* `<*>` indicates variable content
* Used to identify **parameters**

---

## 5. BIO Tag Generation

### Tagging Scheme

* `B-TIME`, `I-TIME`
* `B-LEVEL`, `I-LEVEL`
* `B-COMPONENT`, `I-COMPONENT`
* `B-PARAM`, `I-PARAM`
* `O` (Other)

---

### Labeling Strategy

* Static template text → `O`
* Variable tokens → entity tags
* Structured elements (timestamps, levels) → semantic tags

---

### Example

| Token | 081109 | 203615 | INFO    | Receiving | block | blk_-160 |
| ----- | ------ | ------ | ------- | --------- | ----- | -------- |
| Tag   | B-TIME | I-TIME | B-LEVEL | O         | O     | B-PARAM  |

---

## 6. Tensor Conversion

### Steps

1. Convert tokens → indices
2. Convert tags → indices
3. Apply padding (`max_seq_len`)
4. Store sequence lengths

---

### Domain Encoding

Each sample includes a domain identifier:

```text
Android = 0, Apache = 1, Hadoop = 2, ...
```

This enables:

* shared learning
* domain-aware adaptation

---

## 7. Data Loader Integration

The processed dataset is consumed by:

```text
src/features/data/loader.py
```

Responsibilities:

* Load tensors efficiently
* Create PyTorch `Dataset` and `DataLoader`
* Support batching and shuffling

---

## ⚙️ Key Design Decisions

### ✔ Unified Representation

All logs are mapped into the same BIO tagging space.

### ✔ Domain-Aware Learning

Domain IDs allow specialization without separate models.

### ✔ Template-Guided Labeling

Leverages ground truth templates for high-quality supervision.

---

## 🧠 Key Contribution

> The data module demonstrates that heterogeneous system logs can be transformed into a unified semantic representation, enabling a single sequence model to generalize across multiple domains.

---

## 🛠️ Implementation Skeleton

```python
def tokenize(log_line):
    """Regex-based tokenizer preserving log structure"""
    pass

def align_with_template(log_tokens, template_tokens):
    """Align tokens with template placeholders"""
    pass

def generate_bio_tags(tokens, template):
    """Generate BIO labels for sequence"""
    pass

def build_vocab(all_logs):
    """Construct shared vocabulary"""
    pass
```

---

## ⚠️ Limitations

* Template alignment assumes availability of ground truth templates
* Rare tokens may be mapped to `<UNK>`
* Extremely long logs require truncation

---

## 🔮 Future Improvements

* Character-level embeddings for unknown tokens
* Subword tokenization (BPE / WordPiece)
* Automated template extraction (without CSV dependency)

---
