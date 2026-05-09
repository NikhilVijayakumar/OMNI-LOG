# src\features\data\processor.py
import re
import torch
from collections import Counter


class LogProcessor:
    def __init__(self, max_seq_len=64):
        self.max_seq_len = max_seq_len
        # Unified Regex for Log Structures
        self.token_pattern = re.compile(
            r'[a-zA-Z0-9_\-\.]+|'  # Alphanumeric, underscores, hyphens, dots
            r'[:\(\)\[\]=]|'  # Structural punctuation
            r'\S+'  # Catch-all for remaining non-whitespace
        )
        """
        example
        2026-04-05 07:56:46 INFO [HDFS.DataNode] Receiving block blk_101
        ['2026-04-05', '07:56:46', 'INFO', '[', 'HDFS.DataNode', ']', 'Receiving', 'block', 'blk_101']
        """

        # Semantic Tag Mapping
        self.tag2idx = {
            "<PAD>": 0, "O": 1,
            "B-TIME": 2, "I-TIME": 3,
            "B-LEVEL": 4, "I-LEVEL": 5,
            "B-COMPONENT": 6, "I-COMPONENT": 7,
            "B-PARAM": 8, "I-PARAM": 9
        }
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}

    def tokenize(self, line):
        """Standardized regex tokenization for OMNI-LOG."""
        return self.token_pattern.findall(line)

    def generate_bio_tags(self, tokens, template_tokens, domain: str = None):
        """
        Aligns tokens with LogHub templates to generate BIO tags.
        Templates use '<*>' to represent parameters.
        """
        tags = ["O"] * len(tokens)
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        _PUNCT = {":", "[", "]", "(", ")", "=", ",", "-", "|"}

        # 1. Heuristics for Time and Level
        for i, token in enumerate(tokens):
            # Time: ISO dates, numeric timestamps, time-of-day, syslog months
            if (re.match(r'^\d{2,4}[-:/]\d{2}[-:/]\d{2,4}$|^\d{6,}$|^\d{2}:\d{2}:\d{2}', token) or
                    token.upper()[:3] in months):
                tags[i] = "B-TIME" if (i == 0 or "TIME" not in tags[i - 1]) else "I-TIME"
            # Extend I-TIME to adjacent time-component tokens (e.g. HH after YYYY-MM-DD)
            elif i > 0 and "TIME" in tags[i - 1] and re.match(r'^\d{2}$', token):
                tags[i] = "I-TIME"
            # Level: full words and single-letter Android logcat levels
            elif token.upper() in ["INFO", "DEBUG", "WARN", "WARNING", "ERROR", "FATAL", "SEVERE",
                                    "NOTICE", "TRACE", "V", "D", "I", "W", "E", "F"]:
                tags[i] = "B-LEVEL"

        # 2. Template Alignment for Parameters (Robust 1:N Matching)
        i, j = 0, 0
        while i < len(tokens) and j < len(template_tokens):
            if template_tokens[j] == "<*>":
                tags[i] = "B-PARAM"
                i += 1
                j_next = j + 1
                if j_next < len(template_tokens):
                    next_static = template_tokens[j_next]
                    while i < len(tokens) and tokens[i] != next_static:
                        tags[i] = "I-PARAM"
                        i += 1
                else:
                    while i < len(tokens):
                        tags[i] = "I-PARAM"
                        i += 1
                j += 1
            else:
                i += 1
                j += 1

        # 3. Tag COMPONENT: O-tagged tokens between B-LEVEL and B-PARAM
        # In all LogHub formats the component (class name, app name, node) lives in this gap.
        level_pos = next((k for k, t in enumerate(tags) if t == "B-LEVEL"), None)
        first_param_pos = next((k for k, t in enumerate(tags) if t == "B-PARAM"), None)

        if level_pos is not None:
            end = first_param_pos if first_param_pos is not None else len(tokens)
            first_comp = True
            for k in range(level_pos + 1, end):
                if tags[k] == "O" and tokens[k] not in _PUNCT:
                    tags[k] = "B-COMPONENT" if first_comp else "I-COMPONENT"
                    first_comp = False

        return tags


    def build_vocab(self, all_tokenized_logs, min_freq=1):
        """Constructs a universal vocabulary across all domains."""
        counter = Counter()
        for tokens in all_tokenized_logs:
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= min_freq:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        print(f"Vocab built. Total unique tokens: {len(self.vocab)}")

    def numericalize(self, tokens, tags):
        """Converts text and tags into padded integer sequences."""
        # Truncate and map to indices
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens[:self.max_seq_len]]
        tag_ids = [self.tag2idx.get(t, self.tag2idx["O"]) for t in tags[:self.max_seq_len]]

        length = len(token_ids)

        # Padding
        padding_len = self.max_seq_len - length
        token_ids += [self.vocab["<PAD>"]] * padding_len
        tag_ids += [self.tag2idx["<PAD>"]] * padding_len

        return torch.tensor(token_ids), torch.tensor(tag_ids), length


# Example Usage Test
if __name__ == "__main__":
    proc = LogProcessor(max_seq_len=10)
    log = "081109 203615 INFO Receiving block blk_-160"
    template = "081109 203615 INFO Receiving block <*>"

    tokens = proc.tokenize(log)
    temp_tokens = proc.tokenize(template)
    tags = proc.generate_bio_tags(tokens, temp_tokens)

    print(f"Tokens: {tokens}")
    print(f"Tags:   {tags}")