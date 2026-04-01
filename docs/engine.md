## 📂 Module 4: `engine` — Execution Pipeline & Batching Controller

---

## 📌 Purpose

The `engine` module orchestrates the complete OMNI-LOG pipeline by integrating the **chunker** and **siamese resolver** into a unified execution flow.

It is responsible for:

* managing **batch processing**
* controlling **throughput vs latency trade-offs**
* generating **structured JSON outputs**

---

## 🎯 Objectives

* Execute end-to-end log processing
* Optimize performance using configurable batching
* Dynamically select parsing strategy (chunker vs siamese)
* Produce standardized structured outputs

---

## 📥 Input

* Raw log file (`*.log`)
* Pre-trained chunker model
* Template library (for siamese resolver)

---

## 📤 Output

Structured JSON records:

```json
{
  "raw": "081109 203615 INFO Receiving block blk_-160",
  "structured": {
    "time": "081109 203615",
    "level": "INFO",
    "params": ["blk_-160"]
  },
  "metadata": {
    "method": "BiLSTM-CRF",
    "confidence": 0.98,
    "template_id": 42
  }
}
```

---

## 🧠 Core Responsibilities

### 1. Pipeline Orchestration

Coordinates all modules:

* data loading
* chunker inference
* siamese fallback
* output formatting

---

### 2. Hyperparameter-Driven Batching

Controls performance via configurable parameters.

---

## ⚙️ Key Parameters

| Parameter              | Description                                  |
| ---------------------- | -------------------------------------------- |
| `batch_size`           | Number of logs processed per inference batch |
| `write_batch_size`     | Number of JSON records per output file       |
| `max_seq_len`          | Maximum token length                         |
| `confidence_threshold` | Chunker confidence cutoff                    |
| `similarity_threshold` | Siamese matching threshold                   |

---

## 🔄 Execution Pipeline

```text
Raw Log File
    ↓
Read in Batches (batch_size)
    ↓
Tokenization & Encoding
    ↓
Bi-LSTM-CRF Chunker
    ↓
Confidence Evaluation
    ↓
 ┌───────────────┐
 │ Confidence OK │ → Direct Structuring
 └───────────────┘
           ↓
 ┌───────────────────────┐
 │ Confidence Low        │
 │ → Siamese Resolver    │
 └───────────────────────┘
           ↓
Template Matching
           ↓
JSON Structuring
           ↓
Batch Writing (write_batch_size)
```

---

## 1. Input Batching

Logs are read incrementally:

```python
batch = read_lines(file, batch_size)
```

### Benefits

* memory efficiency
* parallel computation
* GPU utilization

---

## 2. Model Inference

Each batch is passed through the chunker:

```python
pred_tags, confidence = chunker(batch)
```

---

## 3. Confidence-Based Routing

A key decision mechanism:

```python
if confidence >= threshold:
    use_chunker_output()
else:
    use_siamese_resolver()
```

---

### Why This Matters

* avoids over-reliance on one model
* improves robustness
* balances speed vs accuracy

---

## 4. JSON Structuring

Transforms predictions into structured format.

### Extracted Fields

* `time`
* `level`
* `component`
* `params`

---

### Example

```python
{
  "time": "081109 203615",
  "level": "INFO",
  "params": ["blk_-160"]
}
```

---

## 5. Output Batching

Results are written in chunks:

```python
if len(buffer) >= write_batch_size:
    write_to_json(buffer)
```

---

### Benefits

* avoids large memory usage
* supports streaming pipelines
* improves I/O efficiency

---

## 📁 Code Structure

```text
src/features/engine/
│
├── pipeline.py        # main execution logic
├── batch_config.py    # parameter configuration
```

---

## ⚡ Performance Optimization

### Batch Size Trade-off

| Batch Size        | Effect                            |
| ----------------- | --------------------------------- |
| Small (e.g., 16)  | Low latency, real-time            |
| Large (e.g., 512) | High throughput, batch processing |

---

### Domain-Specific Tuning

* Spark / Hadoop → large batches
* Windows / Linux → small batches

---

## 🧠 Key Design Decisions

### ✔ Configurable Batching

Enables flexible deployment scenarios.

### ✔ Hybrid Inference Strategy

Combines chunker and siamese for robustness.

### ✔ Streaming-Friendly Design

Supports large-scale log processing.

### ✔ Structured Output Format

Facilitates downstream analytics.

---

## 🧠 Key Contribution

> The engine module transforms OMNI-LOG from a collection of models into a scalable, configurable system capable of processing large-scale heterogeneous logs with controllable performance characteristics.

---

## ⚠️ Limitations

* Threshold tuning required for optimal performance
* Large batch sizes increase memory usage
* Real-time performance depends on hardware

---

## 🔮 Future Improvements

* Asynchronous pipeline execution
* Distributed processing (Spark / Ray)
* Adaptive batch sizing
* Streaming integration (Kafka)

---

## 🛠️ Implementation Skeleton

```python
class Pipeline:
    def __init__(self, chunker, resolver, config):
        self.chunker = chunker
        self.resolver = resolver
        self.config = config

    def process_logs(self, file_path):
        for batch in read_batches(file_path, self.config.batch_size):
            preds, conf = self.chunker(batch)

            results = []
            for log, pred, c in zip(batch, preds, conf):
                if c >= self.config.conf_threshold:
                    result = build_json(log, pred)
                else:
                    result = self.resolver.resolve(log)

                results.append(result)

            write_batch(results)
```

---
