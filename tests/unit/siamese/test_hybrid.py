import torch
from features.chunker.model import BiLSTM_CRF
from features.siamese.encoder import LogEncoder
from features.data.processor import LogProcessor
from features.siamese.resolver import TemplateResolver
from features.siamese.hybrid_logic import HybridParser


VOCAB_SIZE = 500
EMBED_DIM = 64
HIDDEN_DIM = 128


def _make_processor():
    proc = LogProcessor(max_seq_len=32)
    proc.vocab = {"<PAD>": 0, "<UNK>": 1, "INFO": 2, "Receiving": 3, "block": 4, "blk_-160": 5}
    proc.idx2tag = {0: "<PAD>", 1: "O", 2: "B-TIME", 3: "I-TIME", 4: "B-LEVEL", 5: "I-LEVEL", 6: "B-COMPONENT", 7: "I-COMPONENT", 8: "B-PARAM", 9: "I-PARAM"}
    return proc


def test_high_confidence_uses_chunker():
    processor = _make_processor()
    tag_to_ix = {"<PAD>": 0, "O": 1, "B-TIME": 2, "I-TIME": 3, "B-LEVEL": 4, "I-LEVEL": 5, "B-COMPONENT": 6, "I-COMPONENT": 7, "B-PARAM": 8, "I-PARAM": 9}

    chunker = BiLSTM_CRF(VOCAB_SIZE, tag_to_ix, EMBED_DIM, HIDDEN_DIM)
    chunker.eval()

    # Mock resolver that should NOT be called
    resolver = None

    parser = HybridParser(chunker, resolver, processor, conf_threshold=0.0)
    result = parser.parse_log("INFO Receiving block blk_-160")
    assert result["method"] == "Bi-LSTM-CRF"
    assert result["status"] == "SUCCESS"
    assert "structured_data" in result
    assert result["confidence"] >= 0.0
    print("High confidence -> chunker path: OK")


def test_low_confidence_fallsback_to_siamese():
    processor = _make_processor()
    tag_to_ix = {"<PAD>": 0, "O": 1, "B-TIME": 2, "I-TIME": 3, "B-LEVEL": 4, "I-LEVEL": 5, "B-COMPONENT": 6, "I-COMPONENT": 7, "B-PARAM": 8, "I-PARAM": 9}

    chunker = BiLSTM_CRF(VOCAB_SIZE, tag_to_ix, EMBED_DIM, HIDDEN_DIM)
    chunker.eval()

    # Build a real resolver with a template library
    encoder = LogEncoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    resolver = TemplateResolver(encoder, processor, device="cpu")
    resolver.build_library({0: "INFO Receiving block <*>"})
    resolver.threshold = 0.0

    parser = HybridParser(chunker, resolver, processor, conf_threshold=1.5)
    result = parser.parse_log("INFO Receiving block blk_-160")
    assert result["method"] == "Siamese-Resolver"
    assert result["status"] == "RESOLVED"
    assert "template_id" in result
    print("Low confidence -> siamese path: OK")


def test_no_match_returns_unknown():
    processor = _make_processor()
    tag_to_ix = {"<PAD>": 0, "O": 1, "B-TIME": 2, "I-TIME": 3, "B-LEVEL": 4, "I-LEVEL": 5, "B-COMPONENT": 6, "I-COMPONENT": 7, "B-PARAM": 8, "I-PARAM": 9}

    chunker = BiLSTM_CRF(VOCAB_SIZE, tag_to_ix, EMBED_DIM, HIDDEN_DIM)
    chunker.eval()

    encoder = LogEncoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    resolver = TemplateResolver(encoder, processor, device="cpu")
    resolver.build_library({0: "INFO Receiving block <*>"})
    resolver.threshold = 1.5

    parser = HybridParser(chunker, resolver, processor, conf_threshold=1.5)
    result = parser.parse_log("INFO Receiving block blk_-160")
    assert result["method"] == "None"
    assert result["status"] == "UNKNOWN_PATTERN"
    print("No match -> UNKNOWN_PATTERN: OK")


def test_empty_log():
    processor = _make_processor()
    tag_to_ix = {"<PAD>": 0, "O": 1, "B-TIME": 2, "I-TIME": 3, "B-LEVEL": 4, "I-LEVEL": 5, "B-COMPONENT": 6, "I-COMPONENT": 7, "B-PARAM": 8, "I-PARAM": 9}

    chunker = BiLSTM_CRF(VOCAB_SIZE, tag_to_ix, EMBED_DIM, HIDDEN_DIM)
    chunker.eval()
    encoder = LogEncoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    resolver = TemplateResolver(encoder, processor, device="cpu")

    parser = HybridParser(chunker, resolver, processor, conf_threshold=0.9)
    result = parser.parse_log("")
    assert result["status"] == "UNKNOWN_PATTERN"
    print("Empty log handling: OK")


if __name__ == "__main__":
    test_high_confidence_uses_chunker()
    test_low_confidence_fallsback_to_siamese()
    test_no_match_returns_unknown()
    test_empty_log()
    print("\nAll hybrid parser tests passed!")
