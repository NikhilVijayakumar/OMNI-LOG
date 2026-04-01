import os
import yaml
import torch
import argparse
from src.features.data.loader import get_dataloader
from src.features.chunker.train import train_model


def load_config(path="src/main/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_data_phase(cfg):
    """PHASE 1: Process raw logs into Tensors and Vocab."""
    print("\n--- 🚀 PHASE 1: DATA PREPROCESSING ---")

    # 1. Create directory
    os.makedirs(cfg['paths']['processed_dir'], exist_ok=True)

    # 2. Run the loader (which builds vocab and mixes domains)
    loader, processor = get_dataloader(
        data_dir=cfg['paths']['raw_logs'],
        batch_size=cfg['train']['batch_size'],
        max_seq_len=cfg['preprocessing']['max_seq_len']
    )

    # 3. Save the Vocabulary (Essential for Module 2 & 4)
    vocab_path = os.path.join(cfg['paths']['processed_dir'], "vocab.pth")
    torch.save(processor.vocab, vocab_path)

    print(f"✅ Vocab saved to: {vocab_path}")
    print(f"✅ Total unique tokens: {len(processor.vocab)}")
    return loader


def run_train_phase(cfg):
    """PHASE 2: Train the Bi-LSTM-CRF Chunker."""
    print("\n--- 🧠 PHASE 2: CHUNKER TRAINING ---")

    model_path = os.path.join(cfg['paths']['model_dir'], "best_model.pth")

    # Pass configuration parameters to the training logic
    train_model(
        data_dir=cfg['paths']['raw_logs'],
        model_save_path=model_path,
        config=cfg  # Pass the full config for hyperparams
    )

    print(f"✅ Training Complete. Model stored at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMNI-LOG Pipeline Controller")
    parser.add_argument("--phase", type=str, choices=["data", "train", "all"], default=None)
    parser.add_argument("--config", type=str, default="src/main/config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Prioritize CLI argument over config file
    active_phase = args.phase if args.phase else config.get('pipeline', {}).get('phase', 'all')
    print(f"Executing Phase: {active_phase}")

    if active_phase in ["data", "all"]:
        run_data_phase(config)

    if active_phase in ["train", "all"]:
        run_train_phase(config)