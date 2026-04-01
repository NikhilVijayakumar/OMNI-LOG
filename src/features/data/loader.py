import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .processor import LogProcessor


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
                # Load from CSV if available
                df = pd.read_csv(temp_file)
                logs = df['Content'].tolist()
                templates = df['EventTemplate'].tolist()
            else:
                # Fallback: Load raw log file
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    logs = [line.strip() for line in f.readlines()]
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