## 📂 Module 2: `chunker` — Bi-LSTM-CRF Sequence Labeling Engine

---

## 📌 Purpose

The `chunker` module implements the core **sequence labeling model** responsible for extracting structured semantic entities from raw log sequences.

It formulates log parsing as a **Named Entity Recognition (NER)** problem and uses a **Bi-directional LSTM with a Conditional Random Field (CRF)** to model both contextual dependencies and valid tag transitions.

---

## 🎯 Objectives

* Learn **semantic structure of logs** across domains
* Predict **BIO-tagged entity sequences**
* Capture **long-range dependencies** in log lines
* Enforce **valid tag transitions** using CRF

---

## 📥 Input

From the `data` module:

```python
{
  "tokens": List[int],     # tokenized log line
  "tags": List[int],       # BIO labels (training only)
  "length": int,           # sequence length
  "domain": int            # domain ID
}
```

---

## 📤 Output

Predicted tag sequence:

```python
["B-TIME", "I-TIME", "B-LEVEL", "O", "B-PARAM", ...]
```

Optionally:

* Confidence score
* Emission scores (for CRF decoding)

---

## 🧠 Model Architecture

```text
Token IDs
   ↓
Embedding Layer
   ↓
Bi-LSTM (Forward + Backward)
   ↓
Linear Layer (Tag Projection)
   ↓
CRF Layer (Sequence Optimization)
   ↓
BIO Tag Sequence
```

---

## 1. Embedding Layer

Transforms token indices into dense vectors:

```math
x_i → e_i ∈ ℝ^d
```

### Features

* Learns semantic similarity between tokens
* Can be:

  * randomly initialized
  * or initialized with pre-trained embeddings (optional)

---

## 2. Bi-Directional LSTM

Processes sequences in both directions:

* **Forward LSTM** → captures past context
* **Backward LSTM** → captures future context

---

### Representation

```math
h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}]
```

---

### Why Bi-LSTM?

Logs often contain patterns like:

* timestamp → level → component → message

Bi-LSTM captures:

* what comes before
* what comes after

---

## 3. Linear Projection Layer

Maps hidden states to tag space:

```math
z_i = W h_i + b
```

Where:

* `z_i` = scores for each BIO tag

---

## 4. CRF Layer (Critical Component)

The CRF layer models **dependencies between tags**.

---

### Problem Without CRF

Softmax may produce invalid sequences:

```text
B-LEVEL → I-TIME ❌ (invalid)
```

---

### CRF Solution

Learns transition probabilities:

```math
score(X, Y) = Σ emission + Σ transition
```

Ensures valid outputs like:

```text
B-TIME → I-TIME → B-LEVEL ✔
```

---

### Decoding

* Uses **Viterbi algorithm** to find optimal tag sequence

---

## 🔄 Training Strategy

### Multi-Domain Training

* Logs from all domains are **mixed in batches**
* Model learns **shared grammar**

---

### Optional Enhancement: Domain Embedding

Each sample includes a domain ID:

```text
Embedding = WordEmbedding + DomainEmbedding
```

This allows:

* shared learning
* domain-specific adaptation

---

## 📊 Loss Function

* **Negative Log-Likelihood (CRF Loss)**

```math
L = -log P(Y|X)
```

Optimizes:

* emission scores
* transition scores

---

## ⚙️ Hyperparameters

| Parameter       | Description           |
| --------------- | --------------------- |
| `embedding_dim` | Token embedding size  |
| `hidden_dim`    | LSTM hidden size      |
| `num_layers`    | Number of LSTM layers |
| `dropout`       | Regularization        |
| `use_crf`       | Enable/disable CRF    |
| `batch_size`    | Training batch size   |

---

## 🧪 Training Pipeline

```text
Processed Data
   ↓
Batch Loader
   ↓
Forward Pass
   ↓
CRF Loss
   ↓
Backpropagation
   ↓
Model Update
```

---

## 📁 Code Structure

```text
src/features/chunker/
│
├── model.py    # Bi-LSTM-CRF architecture
├── train.py    # training loop & optimization
```

---

## 🧠 Key Design Decisions

### ✔ Sequence Labeling Formulation

Transforms log parsing into a standard NLP task (NER).

### ✔ Bi-LSTM for Context

Captures both preceding and succeeding token dependencies.

### ✔ CRF for Structural Validity

Ensures grammatically correct tag sequences.

### ✔ Multi-Domain Learning

Single model generalizes across all log categories.

---

## 🧠 Key Contribution

> The chunker demonstrates that system logs from diverse domains share a latent sequential structure that can be effectively modeled using a Bi-LSTM-CRF architecture trained on a unified labeling scheme.

---

## ⚠️ Limitations

* Performance depends on quality of BIO labels
* Long sequences may require truncation
* CRF increases computational cost

---

## 🔮 Future Improvements

* Character-level CNN for handling rare tokens
* Transformer-based encoder (e.g., BERT)
* Attention mechanisms for long-range dependencies

---

## 🛠️ Implementation Skeleton

```python
class BiLSTMChunker(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()
        # embedding layer
        # Bi-LSTM
        # linear layer
        # CRF layer

    def forward(self, x, lengths):
        # return tag scores or decoded sequence
        pass
```

---
