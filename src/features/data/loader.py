# src\features\data\loader.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from features.data.processor import LogProcessor


class LogDataset(Dataset):
    def __init__(self, logs, templates, domain_id, processor):
        """
        Args:
            logs (list): List of raw log strings.
            templates (list): List of corresponding template strings.
            domain_id (int): Integer representing the log domain.
            processor (LogProcessor): Initialized processor for tokenization.
        """
        self.logs = logs
        self.templates = templates
        self.domain_id = domain_id
        self.processor = processor

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log_line = str(self.logs[idx])
        template_line = str(self.templates[idx])

        # 1. Tokenize both
        tokens = self.processor.tokenize(log_line)
        temp_tokens = self.processor.tokenize(template_line)

        # 2. Generate Tags
        tags = self.processor.generate_bio_tags(tokens, temp_tokens)

        # 3. Numericalize to Tensors
        token_ids, tag_ids, length = self.processor.numericalize(tokens, tags)

        return {
            "tokens": token_ids,
            "tags": tag_ids,
            "length": length,
            "domain": torch.tensor(self.domain_id, dtype=torch.long)
        }


def get_dataloader(data_dir, batch_size=32, max_seq_len=64):
    """
    Orchestrates the loading of all *_2k.log files from the data directory.
    """
    processor = LogProcessor(max_seq_len=max_seq_len)
    all_samples = []

    # 1. Detect Domains from files (*_2k.log)
    log_files = [f for f in os.listdir(data_dir) if f.endswith("_2k.log")]
    domains = sorted([f.replace("_2k.log", "") for f in log_files])
    domain_map = {name: i for i, name in enumerate(domains)}

    # 2. Collect files for Vocabulary building
    all_tokenized_logs = []
    dataset_configs = []

    for domain in domains:
        log_file = os.path.join(data_dir, f"{domain}_2k.log")
        temp_file = os.path.join(data_dir, f"{domain}_2k.log_templates.csv")

        if os.path.exists(log_file):
            if os.path.exists(temp_file):
                df = pd.read_csv(temp_file)

                # 1. Identify the 'Logs' column (The raw message)
                possible_content = ['Content', 'LogMessage', 'OriginalLog']
                content_col = next((c for c in possible_content if c in df.columns), None)

                if content_col:
                    template_col = 'EventTemplate' if 'EventTemplate' in df.columns else df.columns[-1]
                    print(f"--- Processing {domain}: Content='{content_col}', Template='{template_col}'")
                    logs = df[content_col].astype(str).tolist()
                    templates = df[template_col].astype(str).tolist()
                else:
                    # Filter out the known non-content columns
                    other_cols = [c for c in df.columns if c.strip() not in ['EventId', 'LineId', 'EventTemplate']]

                    if other_cols:
                        content_col = other_cols[0]
                        template_col = 'EventTemplate' if 'EventTemplate' in df.columns else df.columns[-1]
                        print(f"--- Processing {domain}: Content='{content_col}', Template='{template_col}'")
                        logs = df[content_col].astype(str).tolist()
                        templates = df[template_col].astype(str).tolist()
                    else:
                        # Match 2000 log lines to unique templates using token overlap
                        print(f"--- Processing {domain}: Token-overlap template matching")
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            raw_logs = [line.strip() for line in f.readlines()]
                        
                        template_col = 'EventTemplate' if 'EventTemplate' in df.columns else df.columns[-1]
                        template_list = df[template_col].astype(str).tolist()
                        
                        # Pre-tokenize templates and build static token sets
                        temp_tokens_list = [processor.tokenize(t) for t in template_list]
                        temp_static_sets = []
                        for tt in temp_tokens_list:
                            static = [t for t in tt if t != '<*' and t != '*>' and t != '<*']
                            temp_static_sets.append(set(static))
                        
                        logs = []
                        templates = []
                        raw_logs = [l for l in raw_logs if l.strip()]
                        for raw_log in raw_logs:
                            log_tokens = set(processor.tokenize(raw_log))
                            best_score = -1
                            best_temp_idx = 0
                            for i, static_set in enumerate(temp_static_sets):
                                if not static_set:
                                    continue
                                score = len(log_tokens & static_set) / len(static_set)
                                if score > best_score:
                                    best_score = score
                                    best_temp_idx = i
                            logs.append(raw_log)
                            templates.append(template_list[best_temp_idx])
            else:
                # Fallback: Load raw log file
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    logs = [line.strip() for line in f if line.strip()]
                templates = logs  # Use logs as templates (no PARAM tags generated)

            # Pre-tokenize for vocab
            for l in logs:
                all_tokenized_logs.append(processor.tokenize(str(l)))

            dataset_configs.append({
                "logs": logs,
                "templates": templates,
                "domain_id": domain_map[domain]
            })


    # 3. Build Universal Vocab
    processor.build_vocab(all_tokenized_logs)

    # 4. Create Unified Dataset
    combined_datasets = []
    for config in dataset_configs:
        combined_datasets.append(LogDataset(
            config["logs"],
            config["templates"],
            config["domain_id"],
            processor
        ))

    full_dataset = torch.utils.data.ConcatDataset(combined_datasets)

    return DataLoader(full_dataset, batch_size=batch_size, shuffle=True), processor


# Test Logic
if __name__ == "__main__":
    # Example path structure: data/logs/Hadoop/Hadoop_2k.log
    loader, proc = get_dataloader("data/logs", batch_size=4)

    batch = next(iter(loader))
    print(f"Batch Tokens Shape: {batch['tokens'].shape}")  # [4, max_seq_len]
    print(f"Batch Domain IDs: {batch['domain']}")