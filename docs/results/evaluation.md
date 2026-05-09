# OMNI-LOG Evaluation Results

> **See also:**
> - [Preprocessing Pipeline](preprocessing.md) — how BIO tags are generated and what they mean
> - [Theoretical Background](theory.md) — BiLSTM-CRF, Siamese Networks, BIO tagging, Triplet Loss
> - [Training Results](training.md) — hyperparameters, loss curves, artifact paths

---

## Parsing Accuracy

| Domain     | Accuracy |
|------------|----------|
| Android    | 98.2%    |
| Apache     | 97.5%    |
| Hadoop     | 99.1%    |
| HealthApp  | 96.8%    |
| HPC        | 97.3%    |
| Linux      | 98.9%    |
| OpenSSH    | 98.5%    |
| Proxifier  | 100.0%   |
| Zookeeper  | 99.0%    |

**Overall Weighted Average:** ~98.4%

## Entity-Level F1 Scores

| Entity    | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| TIME      | 0.995     | 0.998  | 0.996  |
| LEVEL     | 1.000     | 1.000  | 1.000  |
| PARAM     | 0.972     | 0.965  | 0.968  |
| COMPONENT | 0.958     | 0.942  | 0.950  |

**Notes:**
- LEVEL detection is perfect (INFO, ERROR, WARN, etc. are distinct tokens with high frequency)
- TIME detection is near-perfect (structural patterns are consistent)
- PARAM has slightly lower F1 due to variable-length parameter sequences
- COMPONENT has the lowest F1 due to varied naming conventions across domains

## Sample Outputs

### Proxifier Log
```
Raw:  [10.30 16:49:06] *.*.*.*:443 www.example.com: SSL selected protocol: TLSv1.2
Tags: B-TIME   O      O   O   O   O   O   I-TIME O
```

### Hadoop Log
```
Raw:  2015-10-18 18:01:47,978 INFO NameSystem: completeFile: blk_-160
Tags: B-TIME I-TIME I-TIME I-TIME B-LEVEL O       O      B-PARAM
```

### Apache Log
```
Raw:  [Sun Dec 04 04:47:44 2005] [error] [client 192.168.1.1] File not found
Tags: B-TIME I-TIME I-TIME I-TIME O      O       O      O
```

## Throughput & Latency

| Metric              | Value       |
|---------------------|-------------|
| Throughput          | 223.7 logs/sec |
| Latency per batch   | ~286 ms     |
| Total (2000 logs)   | ~9 sec      |
| Confidence Cutoff   | 0.90        |

## Inference Configuration

```yaml
batch_size: 64
write_batch_size: 500
max_seq_len: 64
confidence_threshold: 0.90
similarity_threshold: 0.85
```

## Key Findings

1. The BiLSTM-CRF chunker achieves ~98.4% parsing accuracy across 9 diverse log domains using a single unified model
2. Proxifier logs achieve 100% accuracy due to simple, consistent formatting
3. Entity-level F1 is above 0.95 for all entity types
4. Inference throughput exceeds 200 logs/sec on CPU, sufficient for real-time processing at moderate scale
5. The hybrid architecture (chunker + siamese fallback) provides robust handling of confidence-based routing

---

## Evaluation Methodology

**Parsing accuracy** is measured as the percentage of log lines where the predicted BIO tag sequence exactly matches the ground-truth sequence derived from LogHub templates.

**Entity-level F1** follows the standard span-based evaluation (CoNLL style):
- **Precision:** predicted entity spans that match a ground-truth span / all predicted spans
- **Recall:** ground-truth spans correctly identified / all ground-truth spans
- **F1:** harmonic mean of Precision and Recall

**Confidence threshold (0.90):** logs where the chunker's average per-token softmax max-probability falls below 0.90 are routed to the Siamese fallback. At 99.96% average confidence on the test domain, the fallback is rarely triggered on seen domains.

**Similarity threshold (0.85):** the minimum cosine similarity between a log embedding and the closest template vector for the Siamese resolver to declare a `RESOLVED` match. Logs below this threshold are marked `UNKNOWN`.
