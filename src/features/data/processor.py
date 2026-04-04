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

    def generate_bio_tags(self, tokens, template_tokens):
        """
        Aligns tokens with LogHub templates to generate BIO tags.
        Templates use '<*>' to represent parameters.
        """
        tags = ["O"] * len(tokens)
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        # 1. Heuristics for Time and Level
        for i, token in enumerate(tokens):
            # Time Heuristics:
            # - ISO-like or LogHub timestamps: 2015-10-18, 18:01:47,978, 081109
            # - Syslog months: Jun, July...
            if (re.match(r'^\d{2,4}[-:/]\d{2}[-:/]\d{2,4}$|^\d{6,}$|^\d{2}:\d{2}:\d{2}', token) or
                token.upper()[:3] in months):
                tags[i] = "B-TIME" if (i == 0 or "TIME" not in tags[i - 1]) else "I-TIME"
            # Level Heuristics:
            elif token.upper() in ["INFO", "DEBUG", "WARN", "ERROR", "FATAL", "SEVERE", "V", "D", "I", "W", "E"]:
                tags[i] = "B-LEVEL"

        # 2. Template Alignment for Parameters (Robust 1:N Matching)
        # We align tokens index (i) and template tokens index (j)
        i, j = 0, 0
        while i < len(tokens) and j < len(template_tokens):
            if template_tokens[j] == "<*>":
                # Parameter found. Tag until we find a match for the NEXT template token.
                # If it's the last template token, tag everything else as PARAM.
                tags[i] = "B-PARAM"
                i += 1
                j_next = j + 1
                if j_next < len(template_tokens):
                    next_static = template_tokens[j_next]
                    while i < len(tokens) and tokens[i] != next_static:
                        tags[i] = "I-PARAM"
                        i += 1
                else:
                    # Last token is <*>, consume everything
                    while i < len(tokens):
                        tags[i] = "I-PARAM"
                        i += 1
                j += 1 # Move to next template token after matching <*>
            else:
                # Static token alignment - just skip as it's already 'O' or tagged by heuristics
                i += 1
                j += 1

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