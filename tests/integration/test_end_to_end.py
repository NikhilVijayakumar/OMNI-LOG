import os
import torch
from features.engine.pipeline import Pipeline
from features.engine.batch_config import BatchConfig
from features.data.processor import LogProcessor
from features.chunker.model import BiLSTM_CRF
from features.siamese.encoder import LogEncoder
from features.siamese.resolver import TemplateResolver
from features.siamese.hybrid_logic import HybridParser
from features.data.constants import TAG_MAP


def test_end_to_end_with_mock_models():
    processor = LogProcessor(max_seq_len=32)
    processor.vocab = {"<PAD>": 0, "<UNK>": 1}
    for i in range(2, 200):
        processor.vocab[f"token_{i}"] = i

    vocab_size = len(processor.vocab)
    tag_to_ix = TAG_MAP

    chunker = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim=32, hidden_dim=64)
    chunker.eval()

    encoder = LogEncoder(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)
    encoder.eval()

    resolver = TemplateResolver(encoder, processor, device="cpu")
    resolver.build_library({0: "Receiving block <*>", 1: "INFO <*> done"})
    resolver.threshold = 0.0

    parser = HybridParser(chunker, resolver, processor, conf_threshold=0.0)

    config = BatchConfig(batch_size=4, write_batch_size=5)

    log_path = "data/logs"
    log_files = [f for f in os.listdir(log_path) if f.endswith("_2k.log")] if os.path.isdir(log_path) else []
    if not log_files:
        print("Skipping end-to-end test: no log files found in data/logs")
        return

    input_file = os.path.join(log_path, log_files[0])
    output_file = "output/test_e2e_output.json"

    pipeline = Pipeline(parser, config)
    stats = pipeline.process_file(input_file, output_file)

    assert stats["total_logs"] > 0
    assert stats["throughput_logs_per_sec"] > 0

    assert os.path.exists(output_file)
    with open(output_file) as f:
        content = f.read().strip()
    assert content.startswith("[")
    assert content.endswith("]")

    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"End-to-end test passed: {stats['total_logs']} logs, "
          f"{stats['throughput_logs_per_sec']:.1f} logs/sec")


if __name__ == "__main__":
    test_end_to_end_with_mock_models()
    print("\nEnd-to-end integration test passed!")
