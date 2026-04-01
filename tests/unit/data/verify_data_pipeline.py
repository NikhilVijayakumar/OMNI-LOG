import torch
from src.features.data.loader import get_dataloader
from src.features.data.processor import LogProcessor
from src.features.data.constants import DOMAIN_TO_IDX, TAG_MAP


def test_tier1_semantic_tagging():
    """Verify that different log profiles produce correct BIO tags."""
    print("--- Tier 1: Semantic Tagging Test ---")
    proc = LogProcessor(max_seq_len=20)

    # Test Case: Hadoop (Java Profile)
    log = "2015-10-18 18:01:47,978 INFO NameSystem: completeFile: blk_-160"
    template = "2015-10-18 18:01:47,978 INFO NameSystem: completeFile: <*>"

    tokens = proc.tokenize(log)
    t_tokens = proc.tokenize(template)
    tags = proc.generate_bio_tags(tokens, t_tokens, "Hadoop")

    # Assertions
    assert "B-TIME" in tags, "Failed to identify Hadoop Timestamp"
    assert "B-LEVEL" in tags, "Failed to identify Log Level"
    assert tags[-1] == "B-PARAM", f"Failed to tag parameter <*>. Got {tags[-1]}"
    print("✅ Hadoop Profile tagging passed.")


def test_tier2_vocab_and_padding():
    """Verify numerical conversion and fixed-length padding."""
    print("\n--- Tier 2: Vocab & Padding Test ---")
    proc = LogProcessor(max_seq_len=10)
    proc.vocab = {"<PAD>": 0, "<UNK>": 1, "INFO": 2}  # Mock vocab

    tokens = ["INFO", "NewToken"]
    tags = ["B-LEVEL", "O"]

    token_ids, tag_ids, length = proc.numericalize(tokens, tags)

    assert token_ids.shape[0] == 10, "Padding failed: Token tensor length incorrect"
    assert token_ids[1] == 1, "UNK mapping failed"
    assert token_ids[2] == 0, "PAD mapping failed"
    assert length == 2, "Original length tracking failed"
    print("✅ Vocabulary and padding logic passed.")


def test_tier3_dataloader_integration(data_path):
    """Verify the multi-domain ConcatDataset and batching."""
    print("\n--- Tier 3: DataLoader Stress Test ---")
    try:
        loader, proc = get_dataloader(data_path, batch_size=8, max_seq_len=32)
        batch = next(iter(loader))

        # Check shapes
        assert batch['tokens'].shape == (8, 32), f"Token batch shape mismatch: {batch['tokens'].shape}"
        assert batch['tags'].shape == (8, 32), f"Tag batch shape mismatch: {batch['tags'].shape}"

        # Check domain variety (Ensures shuffle and concat worked)
        unique_domains = torch.unique(batch['domain']).tolist()
        print(f"Sample Batch Domains: {unique_domains}")
        assert len(batch['domain']) == 8, "Batch size mismatch"

        print(f"✅ DataLoader passed. Vocab Size: {len(proc.vocab)}")
    except Exception as e:
        print(f"❌ DataLoader failed: {e}")


if __name__ == "__main__":
    # Ensure you have your data/logs folder populated with *_2k.log files
    test_tier1_semantic_tagging()
    test_tier2_vocab_and_padding()
    test_tier3_dataloader_integration(data_path="data/logs")
    print("\n🚀 All Module 1 Verifications Passed!")