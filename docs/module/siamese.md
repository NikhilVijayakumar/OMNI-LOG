## 📂 Module 3: `siamese` — Template Similarity Resolver

---

## 📌 Purpose

The `siamese` module enhances the robustness of the pipeline by resolving **ambiguous or unseen log patterns** using **metric learning**.

Instead of relying solely on sequence labeling, this module maps logs into a **vector space** and retrieves the most similar known template.

---

## 🎯 Objectives

* Handle **unseen log formats**
* Improve parsing when chunker confidence is low
* Learn **semantic similarity between log messages**
* Enable **template-based generalization** across domains

---

## 📥 Input

* Raw or tokenized log sequence
* (Optional) chunker confidence score

---

## 📤 Output

```python id="n2g3t1"
{
  "template_id": int,
  "template_text": str,
  "similarity_score": float
}
```

---

## 🧠 Core Idea: Metric Learning

The Siamese network does not classify logs—it **compares them**.

It learns a function:

```math id="0d4h8x"
f(x) → ℝ^d
```

that maps logs into a vector space where:

* Similar logs → **close vectors**
* Different logs → **far apart**

---

## 🏗️ Architecture

```text id="wq5wz2"
        Log A ──► Encoder ──► Vector A
                      │
                      │ (shared weights)
                      │
        Log B ──► Encoder ──► Vector B

Similarity = cosine(Vector A, Vector B)
```

---

## 1. Bi-LSTM Encoder

The encoder converts a log sequence into a fixed-size vector.

### Pipeline

```text id="g6n7kq"
Token IDs
   ↓
Embedding Layer
   ↓
Bi-LSTM
   ↓
Pooling Layer
   ↓
Fixed-Length Vector
```

---

### Pooling Strategies

* **Mean pooling** (recommended)
* Last hidden state
* Max pooling

---

## 2. Template Library

A precomputed database of known log templates.

---

### Construction

1. Load templates from dataset
2. Encode each template using the encoder
3. Store vectors

---

### Representation

```python id="u3l5oq"
{
  "template_id": int,
  "template_text": str,
  "vector": Tensor,
  "domain": int
}
```

---

## 3. Similarity Computation

Similarity is measured using **cosine similarity**:

```math id="dmpq8w"
sim(A, B) = \frac{A \cdot B}{||A|| ||B||}
```

---

## 4. Resolution Logic

```text id="h7m1pf"
Input Log
   ↓
Encode → Vector
   ↓
Compare with Template Library
   ↓
Find Best Match
   ↓
If similarity ≥ threshold:
        Assign Template
Else:
        Mark as New Pattern
```

---

## 🔄 Training Strategy

### Training Data

* (log, correct_template) pairs
* negative samples from other templates

---

### Loss Functions

#### Contrastive Loss

```math id="1c6p2l"
L = y * d^2 + (1 - y) * max(0, margin - d)^2
```

---

#### Triplet Loss (Preferred)

```math id="5j8n2z"
L = max(0, d(a,p) - d(a,n) + margin)
```

Where:

* `a` = anchor (log)
* `p` = positive (correct template)
* `n` = negative (wrong template)

---

## 🔗 Integration with Chunker

The resolver is triggered conditionally:

```text id="dcz7y9"
If Chunker Confidence ≥ threshold:
    Use Chunker Output
Else:
    Use Siamese Resolver
```

---

## 📁 Code Structure

```text id="i8tq1k"
src/features/siamese/
│
├── encoder.py     # Bi-LSTM encoder
├── resolver.py    # similarity + template matching
```

---

## ⚙️ Key Hyperparameters

| Parameter              | Description          |
| ---------------------- | -------------------- |
| `embedding_dim`        | Token embedding size |
| `hidden_dim`           | Encoder LSTM size    |
| `margin`               | Triplet loss margin  |
| `similarity_threshold` | Matching cutoff      |
| `pooling_type`         | mean / max / last    |

---

## 🧠 Key Design Decisions

### ✔ Shared Encoder

Same encoder processes logs and templates → consistent representation

### ✔ Vector Space Matching

Avoids rigid rule-based parsing

### ✔ Threshold-Based Resolution

Controls precision vs recall tradeoff

### ✔ Template Reuse

Leverages structured knowledge from dataset

---

## 🧠 Key Contribution

> The Siamese module enables OMNI-LOG to generalize beyond seen training data by mapping logs into a semantic vector space and performing similarity-based template retrieval.

---

## ⚠️ Limitations

* Requires template library maintenance
* Similarity threshold tuning is critical
* Performance depends on embedding quality

---

## 🔮 Future Improvements

* Replace Bi-LSTM encoder with Transformer
* Use FAISS for fast similarity search
* Dynamic template expansion
* Hybrid scoring (chunker + similarity fusion)

---

## 🛠️ Implementation Skeleton

```python id="i4o7lx"
class SiameseResolver(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, log_a, log_b):
        vec_a = self.encoder(log_a)
        vec_b = self.encoder(log_b)
        return cosine_similarity(vec_a, vec_b)
```

---
