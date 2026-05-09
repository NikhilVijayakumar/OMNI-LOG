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
    print("\n--- [PHASE 1] DATA PREPROCESSING ---")
    os.makedirs(cfg['paths']['processed_dir'], exist_ok=True)

    loader, processor = get_dataloader(
        data_dir=cfg['paths']['raw_logs'],
        batch_size=cfg['train']['batch_size'],
        max_seq_len=cfg['preprocessing']['max_seq_len']
    )

    vocab_path = os.path.join(cfg['paths']['processed_dir'], "vocab.pth")
    torch.save(processor.vocab, vocab_path)

    print(f"[OK] Vocab saved: {vocab_path} ({len(processor.vocab)} tokens)")
    return loader  # Return loader to use in Phase 2 if needed


def run_train_phase(cfg, loader=None):
    from features.chunker.train import train_model
    from features.siamese.train_siamese import train_siamese, TripletLogDataset
    from features.siamese.encoder import LogEncoder
    from torch.utils.data import DataLoader
    from features.data.loader import get_dataloader

    print("\n--- [PHASE 2] TRAINING (CHUNKER & SIAMESE) ---")
    data_dir = cfg['paths']['raw_logs']
    model_save_path = os.path.join(cfg['paths']['model_dir'], "best_model.pth")
    
    # 1. Train Chunker
    print("Training Bi-LSTM-CRF Chunker...")
    train_model(data_dir=data_dir, model_save_path=model_save_path, config=cfg)
    
    # 2. Train Siamese (Optional or full)
    print("Training Siamese Encoder...")
    loader, processor = get_dataloader(data_dir, batch_size=cfg['train']['batch_size'], max_seq_len=cfg['preprocessing']['max_seq_len'])
    
    all_logs = []
    all_templates = []
    for dataset in loader.dataset.datasets:
        all_logs.extend(dataset.logs)
        all_templates.extend(dataset.templates)
        
    triplet_dataset = TripletLogDataset(all_logs, all_templates, processor)
    siamese_loader = DataLoader(triplet_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
    
    encoder = LogEncoder(
        vocab_size=len(processor.vocab),
        embedding_dim=cfg['train']['embedding_dim'],
        hidden_dim=cfg['train']['hidden_dim']
    )
    
    trained_encoder = train_siamese(encoder, siamese_loader, epochs=cfg['train']['epochs'])
    
    siamese_save_path = os.path.join(cfg['paths']['model_dir'], "siamese_encoder.pth")
    os.makedirs(os.path.dirname(siamese_save_path), exist_ok=True)
    torch.save(trained_encoder.state_dict(), siamese_save_path)
    print(f"[OK] Siamese model saved: {siamese_save_path}")

def run_inference_phase(cfg):
    from features.engine.pipeline import Pipeline
    from features.engine.batch_config import BatchConfig
    from features.siamese.hybrid_logic import HybridParser
    from features.chunker.model import BiLSTM_CRF
    from features.siamese.resolver import TemplateResolver
    from features.siamese.encoder import LogEncoder
    from features.data.processor import LogProcessor
    from features.data.constants import TAG_MAP
    import json
    import glob
    import pandas as pd

    print("\n--- [PHASE 3] INFERENCE PIPELINE ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vocab
    vocab_path = os.path.join(cfg['paths']['processed_dir'], "vocab.pth")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError("Vocab not found. Run 'data' phase first.")
    vocab = torch.load(vocab_path, map_location=device)
    
    processor = LogProcessor(max_seq_len=cfg['preprocessing']['max_seq_len'])
    processor.vocab = vocab
    
    # 2. Load Chunker
    print("Loading Models...")
    chunker = BiLSTM_CRF(
        vocab_size=len(vocab),
        tag_to_ix=TAG_MAP,
        embedding_dim=cfg['train']['embedding_dim'],
        hidden_dim=cfg['train']['hidden_dim']
    ).to(device)
    
    model_save_path = os.path.join(cfg['paths']['model_dir'], "best_model.pth")
    if os.path.exists(model_save_path):
        saved = torch.load(model_save_path, map_location=device)
        chunker.load_state_dict(saved.get('model_state_dict', saved))
    chunker.eval()
    
    # 3. Load Siamese Encoder
    encoder = LogEncoder(
        vocab_size=len(vocab),
        embedding_dim=cfg['train']['embedding_dim'],
        hidden_dim=cfg['train']['hidden_dim']
    ).to(device)
    siamese_save_path = os.path.join(cfg['paths']['model_dir'], "siamese_encoder.pth")
    if os.path.exists(siamese_save_path):
        encoder.load_state_dict(torch.load(siamese_save_path, map_location=device))
        
    resolver = TemplateResolver(encoder=encoder, processor=processor, device=device)
    
    # 4. Build Template Library
    print("Building Template Library...")
    data_dir = cfg['paths']['raw_logs']
    template_files = glob.glob(os.path.join(data_dir, "*_templates.csv"))
    templates_dict = {}
    idx = 0
    for file in template_files:
        df = pd.read_csv(file)
        if 'EventTemplate' in df.columns:
            for t in df['EventTemplate'].dropna().unique():
                templates_dict[f"T{idx}"] = str(t)
                idx += 1
    
    if templates_dict:
        resolver.build_library(templates_dict)
    
    # 5. Initialize Pipeline
    hybrid_parser = HybridParser(chunker=chunker, resolver=resolver, processor=processor, conf_threshold=0.90)
    batch_config = BatchConfig.from_yaml("src/main/config/config.yaml")
    pipeline = Pipeline(hybrid_parser=hybrid_parser, config=batch_config)
    
    # 6. Process Logs
    log_files = glob.glob(os.path.join(data_dir, "*_2k.log"))
    if not log_files:
        print("No log files found to process.")
        return
        
    input_file = log_files[0]
    output_file = cfg['paths']['output_json']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Starting pipeline on: {input_file}")
    stats = pipeline.process_file(input_file, output_file)
    
    print("\n[OK] Inference Complete!")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    config = load_config("src/main/config/config.yaml")
    active_phase = config.get("pipeline", {}).get("phase", "all")
    data_loader = None

    # 1. Run Data Preprocessing (Module 1)
    if active_phase in ["data", "all"]:
        data_loader = run_data_phase(config)

    # 2. Run Training Phase (Module 2)
    if active_phase in ["train", "all"]:
        run_train_phase(config, data_loader)
        
    # 3. Run Inference Phase (Module 3)
    if active_phase in ["inference", "all"]:
        run_inference_phase(config)


