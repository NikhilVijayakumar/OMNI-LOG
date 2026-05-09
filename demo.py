"""
OMNI-LOG Demo: Run the trained pipeline against any log file.

Usage:
    python demo.py                                   # uses first log file found in data/logs/
    python demo.py --input data/logs/Hadoop_2k.log
    python demo.py --input data/logs/Linux_2k.log --output output/json/linux_structured.json
"""

import os
import sys
import glob
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="OMNI-LOG inference demo")
    parser.add_argument("--input",  default=None, help="Path to a .log file (default: first *_2k.log in data/logs/)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: output/json/<domain>_structured.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("  OMNI-LOG: Cross-Domain Log Structuring Pipeline")
    print("=" * 60)

    # 1. Resolve input file
    if args.input:
        input_file = args.input
        if not os.path.exists(input_file):
            print(f"[ERROR] File not found: {input_file}")
            sys.exit(1)
    else:
        data_dir = "data/logs"
        log_files = sorted(glob.glob(os.path.join(data_dir, "*_2k.log")))
        if not log_files:
            print(f"[ERROR] No *_2k.log files found in '{data_dir}'. Pass --input <file>.")
            sys.exit(1)
        input_file = log_files[0]

    domain = os.path.basename(input_file).replace("_2k.log", "").replace(".log", "")
    output_file = args.output or f"output/json/{domain}_structured.json"

    print(f"\n[1] Input:  {input_file}")
    print(f"    Output: {output_file}")

    # 2. Check for trained models
    chunker_path = "output/models/chunker/best_model.pth"
    siamese_path = "output/models/chunker/siamese_encoder.pth"
    vocab_path   = "output/processed/vocab.pth"

    if not os.path.exists(chunker_path) or not os.path.exists(siamese_path):
        print("\n[2] Trained models not found — running training pipeline first...")
        ret = os.system(f"{sys.executable} src/main/main.py")
        if ret != 0:
            print("[ERROR] Training pipeline failed.")
            sys.exit(1)
        print("[OK] Training complete.")
    else:
        print(f"\n[2] Trained models found.")
        print(f"     Chunker: {chunker_path}  ({os.path.getsize(chunker_path)//1024} KB)")
        print(f"     Siamese: {siamese_path}  ({os.path.getsize(siamese_path)//1024} KB)")

    # 3. Load models
    print("\n[3] Loading models...")
    from features.engine.pipeline import Pipeline
    from features.engine.batch_config import BatchConfig
    from features.data.processor import LogProcessor
    from features.chunker.model import BiLSTM_CRF
    from features.siamese.encoder import LogEncoder
    from features.siamese.resolver import TemplateResolver
    from features.siamese.hybrid_logic import HybridParser
    from features.data.constants import TAG_MAP

    # Load chunker checkpoint (saved as dict with model_state_dict, vocab, tag_map)
    checkpoint = torch.load(chunker_path, map_location="cpu", weights_only=False)
    vocab    = checkpoint["vocab"]
    tag_to_ix = checkpoint.get("tag_map", TAG_MAP)
    vocab_size = len(vocab)

    emb_dim    = 128
    hidden_dim = 256

    chunker = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim=emb_dim, hidden_dim=hidden_dim)
    chunker.load_state_dict(checkpoint["model_state_dict"])
    chunker.eval()
    print(f"     Chunker loaded  — {vocab_size} vocab tokens, {sum(p.numel() for p in chunker.parameters()):,} params")

    # Load siamese encoder (saved as plain state_dict via torch.save(encoder.state_dict(), ...))
    encoder = LogEncoder(vocab_size=vocab_size, embedding_dim=emb_dim, hidden_dim=hidden_dim)
    encoder.load_state_dict(torch.load(siamese_path, map_location="cpu", weights_only=False))
    encoder.eval()
    print(f"     Encoder loaded  — {sum(p.numel() for p in encoder.parameters()):,} params")

    # 4. Build processor + template library from available template CSVs
    processor = LogProcessor(max_seq_len=64)
    processor.vocab = vocab

    resolver = TemplateResolver(encoder, processor, device="cpu")

    import pandas as pd
    data_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else "data/logs"
    template_files = glob.glob(os.path.join(data_dir, "*_templates.csv"))
    templates_dict = {}
    for tf in template_files:
        df = pd.read_csv(tf)
        if "EventTemplate" in df.columns:
            for t in df["EventTemplate"].dropna().unique():
                templates_dict[f"T{len(templates_dict)}"] = str(t)

    if templates_dict:
        resolver.build_library(templates_dict)
        print(f"     Template library — {len(templates_dict)} templates from {len(template_files)} domain(s)")
    else:
        print("     No template CSVs found — Siamese resolver will have no library (OK for demo)")

    # 5. Run pipeline
    print(f"\n[4] Running pipeline on {os.path.basename(input_file)}...")
    hybrid_parser = HybridParser(chunker=chunker, resolver=resolver, processor=processor, conf_threshold=0.90)
    config   = BatchConfig(batch_size=64, write_batch_size=500)
    pipeline = Pipeline(hybrid_parser=hybrid_parser, config=config)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    stats = pipeline.process_file(input_file, output_file)

    # 6. Display results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Input file:   {os.path.basename(input_file)}")
    print(f"  Total logs:   {stats['total_logs']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Throughput:   {stats['throughput_logs_per_sec']:.1f} logs/sec")
    print(f"  Total time:   {stats['total_time_sec']:.2f} sec")
    print(f"  Output:       {output_file}")

    # 7. Show sample records
    import json
    with open(output_file) as f:
        records = json.load(f)

    print("\n  --- Sample output (first 3 records) ---")
    for r in records[:3]:
        print(json.dumps(r, indent=4))

    print("=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
