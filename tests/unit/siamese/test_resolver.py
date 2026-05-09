import torch
from features.siamese.encoder import LogEncoder
from features.data.processor import LogProcessor
from features.siamese.resolver import TemplateResolver


VOCAB_SIZE = 500
EMBED_DIM = 64
HIDDEN_DIM = 128


def _make_encoder():
    return LogEncoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)


def _make_processor():
    proc = LogProcessor(max_seq_len=32)
    proc.vocab = {"<PAD>": 0, "<UNK>": 1, "Receiving": 2, "block": 3, "blk_-160": 4, "from": 5, "INFO": 6}
    for i in range(7, 100):
        proc.vocab[f"token_{i}"] = i
    return proc


def test_build_library():
    encoder = _make_encoder()
    processor = _make_processor()
    resolver = TemplateResolver(encoder, processor, device="cpu")

    templates = {
        0: "Receiving block <*> from <*>",
        1: "INFO Receiving block <*>",
    }
    resolver.build_library(templates)
    assert resolver.template_vectors is not None
    assert resolver.template_vectors.shape[0] == 2
    assert resolver.template_metadata["ids"] == [0, 1]
    print("Build library: OK")


def test_resolve_match():
    encoder = _make_encoder()
    processor = _make_processor()
    resolver = TemplateResolver(encoder, processor, device="cpu")

    templates = {0: "Receiving block <*> from <*>"}
    resolver.build_library(templates)

    # Set threshold to 0.0 so any similarity counts as match
    resolver.threshold = 0.0

    result = resolver.resolve("Receiving block blk_-160 from 10.0.0.1")
    assert result["match_found"] is True
    assert result["template_id"] == 0
    assert result["similarity"] >= 0.0
    print("Resolve match: OK")


def test_resolve_no_match():
    encoder = _make_encoder()
    processor = _make_processor()
    resolver = TemplateResolver(encoder, processor, device="cpu")

    templates = {0: "INFO Receiving block <*>"}
    resolver.build_library(templates)

    # Set threshold high so nothing matches
    resolver.threshold = 1.5

    result = resolver.resolve("something completely different")
    assert result["match_found"] is False
    print("Resolve no match: OK")


def test_threshold_filtering():
    encoder = _make_encoder()
    processor = _make_processor()
    resolver = TemplateResolver(encoder, processor, device="cpu")

    templates = {0: "INFO Receiving block <*>"}
    resolver.build_library(templates)

    # With threshold 0.0, everything matches
    resolver.threshold = 0.0
    result = resolver.resolve("INFO Receiving block blk_-160")
    assert result["match_found"] is True

    # With threshold > 1.0, nothing matches
    resolver.threshold = 1.5
    result = resolver.resolve("INFO Receiving block blk_-160")
    assert result["match_found"] is False
    print("Threshold filtering: OK")


if __name__ == "__main__":
    test_build_library()
    test_resolve_match()
    test_resolve_no_match()
    test_threshold_filtering()
    print("\nAll resolver tests passed!")
