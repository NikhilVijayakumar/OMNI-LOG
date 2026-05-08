import os
import yaml
import torch
import argparse
from features.data.loader import get_dataloader



def load_config(path="src/main/config/config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_data_phase(cfg):
    print("\n--- 🚀 PHASE 1: DATA PREPROCESSING ---")
    os.makedirs(cfg['paths']['processed_dir'], exist_ok=True)

    loader, processor = get_dataloader(
        data_dir=cfg['paths']['raw_logs'],
        batch_size=cfg['train']['batch_size'],
        max_seq_len=cfg['preprocessing']['max_seq_len']
    )

    vocab_path = os.path.join(cfg['paths']['processed_dir'], "vocab.pth")
    torch.save(processor.vocab, vocab_path)

    print(f"✅ Vocab saved: {vocab_path} ({len(processor.vocab)} tokens)")
    return loader  # Return loader to use in Phase 2 if needed


def run_train_phase(cfg, loader=None):
    pass


if __name__ == "__main__":
    config = load_config("src/main/config/config.yaml")
    active_phase = config.get("pipeline", {}).get("phase", "all")
    data_loader = None

    # 1. Run Data Preprocessing (Module 1)
    if active_phase in ["data", "all"]:
        data_loader = run_data_phase(config)

