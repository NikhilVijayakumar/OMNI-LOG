# OMNI-LOG Preprocessing Pipeline

## Overview

Preprocessing converts raw log files into labeled, padded integer tensors ready for model training.

**Artifact produced:** `output/processed/vocab.pth`

---

## Pipeline Artifact Map

```
[Raw Logs: data/logs/*_2k.log]
         |
         v
  [Phase 1 — PREPROCESSING]
  src/features/data/processor.py
  src/features/data/loader.py
         |
         |-- output/processed/vocab.pth   <-- Universal vocabulary
         v
  [Phase 2 — TRAINING]
  src/features/chunker/train.py
  src/features/siamese/train_siamese.py
         |
         |-- output/models/chunker/best_model.pth
         |-- output/models/chunker/siamese_encoder.pth
         v
  [Phase 3 — INFERENCE]
  src/features/engine/pipeline.py
         |
         |-- output/json/<domain>.jsonl   <-- Structured parsed logs
```

---

## Step 1 — Data Input Format (LogHub)

Each domain in the dataset consists of two files:

| File | Purpose |
|------|---------|
| `<Domain>_2k.log` | 2,000 raw log lines per domain |
| `<Domain>_2k.log_templates.csv` | Corresponding log templates with `<*>` for variable fields |

**Template example:**
```
Raw log:    Receiving block blk_-160 from /10.250.14.224
Template:   Receiving block <*> from <*>
```

The `<*>` wildcard marks positions that hold variable parameters (IPs, block IDs, file paths, etc.).

**Domains covered:** Android, Apache, Hadoop, HealthApp, HPC, Linux, OpenSSH, Proxifier, Zookeeper

**Domain-specific log profiles** (`constants.py`):

| Profile | Domains | Example Format |
|---------|---------|----------------|
| syslog | Linux, OpenSSH | `Jun 14 15:16:01` |
| java_bigdata | Hadoop, Zookeeper | `2015-10-18 18:01:47,978` |
| apache | Apache | `[Sun Dec 04 04:47:44 2005]` |
| proxifier | Proxifier | `[10.30 16:49:06]` |
| android | Android | `03-17 16:13:38.811` |
| healthapp | HealthApp | `20171223-22:15:29:606\|` |
| hpc | HPC | `134681 node-246` |

---

## Step 2 — Tokenization

**Class:** `LogProcessor.tokenize()` in `src/features/data/processor.py`

A unified regex pattern splits log lines into meaningful tokens while preserving structure:

```python
r'[a-zA-Z0-9_\-\.]+'  # Words, IDs, IPs, paths (e.g. blk_-160, 10.0.0.1)
r'|[:\(\)\[\]=]'       # Structural punctuation (kept as separate tokens)
r'|\S+'                # Catch-all for any remaining non-whitespace
```

**Example:**
```
Input:  "2026-04-05 07:56:46 INFO [HDFS.DataNode] Receiving block blk_101"
Output: ['2026-04-05', '07:56:46', 'INFO', '[', 'HDFS.DataNode', ']',
         'Receiving', 'block', 'blk_101']
```

---

## Step 3 — BIO Tag Generation

**Class:** `LogProcessor.generate_bio_tags()` in `src/features/data/processor.py`

BIO (Beginning–Inside–Outside) tags label each token with its semantic role. Tag generation follows a **3-step heuristic + alignment pipeline**.

### The BIO Tag Schema

| Tag | Meaning |
|-----|---------|
| `B-TIME` | First token of a timestamp |
| `I-TIME` | Continuation of a timestamp |
| `B-LEVEL` | Log severity level (INFO, ERROR, ...) |
| `I-LEVEL` | Continuation of a level token |
| `B-COMPONENT` | First token of a component name |
| `I-COMPONENT` | Continuation of a component name |
| `B-PARAM` | First token of a variable parameter |
| `I-PARAM` | Continuation of a variable parameter |
| `O` | Non-entity / outside any named span |
| `<PAD>` | Padding token (no semantic meaning) |

### Step 3a — Time and Level Heuristics

Regex-based rules detect timestamps and severity levels without needing a template:

- **B-TIME / I-TIME:** Matches ISO dates (`2015-10-18`), numeric timestamps (`081109`), time-of-day (`07:56:46`), and syslog months (`Jun`, `Dec`, etc.)
- **B-LEVEL:** Matches exact severity words: `INFO`, `DEBUG`, `WARN`, `WARNING`, `ERROR`, `FATAL`, `SEVERE`, `NOTICE`, `TRACE`, plus single-letter Android logcat levels (`V`, `D`, `I`, `W`, `E`, `F`)

### Step 3b — Template Alignment for Parameters

The raw log is aligned token-by-token against the LogHub template. When a `<*>` wildcard is encountered, tokens are tagged `B-PARAM` / `I-PARAM` until the next static template word is matched. This handles 1:N mapping (one wildcard spanning multiple tokens).

```
Template: Receiving block <*> from <*>
Tokens:   Receiving  block  blk_101     from  /10.0.0.1
Tags:     O          O      B-PARAM     O     B-PARAM
```

For domains without a content column, template matching falls back to **token-overlap scoring** — each raw log is matched to the template with the highest Jaccard-style token overlap against static (non-wildcard) template tokens.

### Step 3c — Component Tagging

Any `O`-tagged token that sits between the `B-LEVEL` position and the first `B-PARAM` position (excluding punctuation) is tagged as a component (class name, module name, node name):

```
Tokens: INFO  [  HDFS.DataNode  ]  Receiving block  blk_101
Tags:   B-LEVEL O  B-COMPONENT  O  O         O     B-PARAM
```

---

## Step 4 — Vocabulary Building

**Class:** `LogProcessor.build_vocab()` in `src/features/data/processor.py`

A universal vocabulary is built from all tokenized logs across all 9 domains in a single pass. A `Counter` accumulates token frequencies; tokens meeting `min_freq >= 1` (default) are added.

**Special tokens always present:**

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Padding to fixed sequence length |
| `<UNK>` | 1 | Out-of-vocabulary token fallback |
| `<SOS>` | 2 | Start-of-sequence marker |
| `<EOS>` | 3 | End-of-sequence marker |

**Result:** `vocab.pth` saved to `output/processed/` — a Python dict mapping `{token: index}`.

**Vocabulary size from training:** 17,067 unique tokens across 9 domains × 2,000 logs.

---

## Step 5 — Numericalization and Padding

**Class:** `LogProcessor.numericalize()` in `src/features/data/processor.py`

Each tokenized + tagged log is converted to fixed-length integer tensors:

1. Map each token to its vocab index (unknown tokens → `<UNK>` index 1)
2. Map each tag to its tag index (from `TAG_MAP` in `constants.py`)
3. Truncate to `max_seq_len = 64` tokens
4. Pad with `<PAD>` (index 0) up to `max_seq_len`

**Output per log:**
- `token_ids`: `torch.LongTensor[64]`
- `tag_ids`: `torch.LongTensor[64]`
- `length`: integer — actual (non-padded) token count

---

## Step 6 — Dataset and DataLoader

**Class:** `LogDataset`, `get_dataloader()` in `src/features/data/loader.py`

Each domain becomes a `LogDataset`. All domain datasets are concatenated into a single `ConcatDataset` and returned as a unified `DataLoader`. Each batch item contains:

```python
{
    "tokens": LongTensor[batch, 64],  # Numericalized log tokens
    "tags":   LongTensor[batch, 64],  # BIO tag indices
    "length": int,                    # Real token count (before padding)
    "domain": LongTensor              # Domain ID (0–8)
}
```

This unified dataset structure means the chunker is trained on all 9 domains simultaneously — a single model learns a cross-domain representation.

---

## Preprocessing Summary

| Stage | Input | Output |
|-------|-------|--------|
| Load | `*_2k.log` + `*_templates.csv` | Raw log strings + template strings |
| Tokenize | Raw log string | List of string tokens |
| BIO Tag | Tokens + template tokens | List of BIO tag strings |
| Vocab Build | All tokenized logs | `vocab.pth` dict (17,067 entries) |
| Numericalize | Tokens + tags | `LongTensor[64]` × 2 + length int |
| DataLoader | `LogDataset` × 9 domains | Batched tensors for training |
