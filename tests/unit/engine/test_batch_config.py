import os
import tempfile
import yaml
from features.engine.batch_config import BatchConfig


def test_default_values():
    cfg = BatchConfig()
    assert cfg.batch_size == 64
    assert cfg.write_batch_size == 500
    assert cfg.max_seq_len == 64
    assert cfg.confidence_threshold == 0.90
    assert cfg.similarity_threshold == 0.85
    print("Default values: OK")


def test_from_yaml_missing_file():
    cfg = BatchConfig.from_yaml("nonexistent.yaml")
    assert cfg.batch_size == 64
    assert cfg.max_seq_len == 64
    print("Missing YAML fallback: OK")


def test_from_yaml_valid():
    yaml_content = """
train:
  batch_size: 32
preprocessing:
  max_seq_len: 128
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        cfg = BatchConfig.from_yaml(tmp_path)
        assert cfg.batch_size == 32
        assert cfg.max_seq_len == 128
        assert cfg.write_batch_size == 500
        assert cfg.confidence_threshold == 0.90
        assert cfg.similarity_threshold == 0.85
        print("Valid YAML loading: OK")
    finally:
        os.unlink(tmp_path)


def test_from_yaml_partial():
    yaml_content = """
train:
  batch_size: 16
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        cfg = BatchConfig.from_yaml(tmp_path)
        assert cfg.batch_size == 16
        assert cfg.max_seq_len == 64
        print("Partial YAML loading: OK")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_default_values()
    test_from_yaml_missing_file()
    test_from_yaml_valid()
    test_from_yaml_partial()
    print("\nAll batch_config tests passed!")
