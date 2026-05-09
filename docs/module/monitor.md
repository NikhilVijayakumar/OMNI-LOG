## 📂 Module 5: `monitor` — Evaluation & Experiment Tracking

---

## 📌 Purpose

The `monitor` module is responsible for evaluating the performance of the OMNI-LOG pipeline and tracking experiments systematically.

It provides:

* **quantitative metrics** for model performance
* **experiment tracking** for reproducibility
* **performance analysis** across configurations

---

## 🎯 Objectives

* Measure parsing and extraction accuracy
* Evaluate template matching performance
* Analyze system efficiency (throughput & latency)
* Track experiments across hyperparameter variations

---

## 📥 Input

* Ground truth labels (from dataset)
* Predicted outputs (from pipeline)
* Runtime logs and performance metrics

---

## 📤 Output

* Evaluation scores (PA, F1, etc.)
* MLflow experiment logs
* Performance plots (optional)

---

## 📊 Evaluation Metrics

### 1. Parsing Accuracy (PA)

Measures the percentage of logs where **all entities are correctly extracted**.

```math id="vfd7tt"
PA = \frac{\text{Correctly Parsed Logs}}{\text{Total Logs}}
```

---

### 2. Entity-Level F1 Score

Evaluates extraction quality for each entity type:

* `TIME`
* `LEVEL`
* `COMPONENT`
* `PARAM`

---

### Formula

```math id="0t8d6m"
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
```

---

### 3. Template Matching Accuracy

Evaluates Siamese resolver:

```math id="n9x0xg"
Accuracy = \frac{\text{Correct Template Matches}}{\text{Total Predictions}}
```

---

### 4. Throughput

Measures processing speed:

```math id="yaxp5x"
Throughput = \frac{\text{Number of Logs}}{\text{Time (seconds)}}
```

---

### 5. Latency

Measures time per batch:

```math id="5b8a5m"
Latency = \frac{\text{Total Processing Time}}{\text{Number of Batches}}
```

---

## 📈 Experiment Tracking (MLflow)

The module integrates with MLflow to track experiments.

---

### What is Logged?

* Hyperparameters:

  * `batch_size`
  * `hidden_dim`
  * `embedding_dim`
  * `use_crf`
* Metrics:

  * Parsing Accuracy
  * F1 Score
  * Throughput
* Artifacts:

  * trained models
  * logs
  * plots

---

### Example Workflow

```python id="z3k3sz"
import mlflow

with mlflow.start_run():
    mlflow.log_param("batch_size", 256)
    mlflow.log_metric("F1_score", 0.91)
```

---

## 📊 Performance Analysis

### Batch Size vs Throughput

The system evaluates how performance changes with different batch sizes.

---

### Expected Trend

| Batch Size | Throughput | Latency |
| ---------- | ---------- | ------- |
| Small      | Low        | Low     |
| Medium     | Medium     | Medium  |
| Large      | High       | High    |

---

## 📁 Code Structure

```text id="c3fd2w"
src/features/monitor/
│
├── metrics.py        # evaluation metrics
├── mlflow_utils.py   # experiment tracking
```

---

## 🧠 Key Design Decisions

### ✔ Multi-Level Evaluation

Evaluates both:

* entity extraction (chunker)
* template matching (siamese)

---

### ✔ System-Level Metrics

Includes:

* throughput
* latency

---

### ✔ Experiment Reproducibility

MLflow ensures:

* consistent tracking
* easy comparison

---

## 🧠 Key Contribution

> The monitor module provides a comprehensive evaluation framework that measures both model accuracy and system efficiency, enabling reproducible and data-driven optimization of the OMNI-LOG pipeline.

---

## ⚠️ Limitations

* Requires labeled ground truth data
* MLflow setup adds overhead
* Performance metrics depend on hardware

---

## 🔮 Future Improvements

* Dashboard visualization (Grafana / Streamlit)
* Automated hyperparameter tuning
* Real-time monitoring integration
* Alerting for performance degradation

---

## 🛠️ Implementation Skeleton

```python id="n8p0k6"
def compute_parsing_accuracy(preds, labels):
    # compare full sequences
    pass

def compute_f1(preds, labels):
    # entity-level evaluation
    pass

def log_experiment(params, metrics):
    # MLflow logging
    pass
```

---
