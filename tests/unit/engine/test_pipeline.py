import os
import tempfile
from features.engine.pipeline import Pipeline
from features.engine.batch_config import BatchConfig
from features.siamese.hybrid_logic import HybridParser


class MockHybridParser:
    def parse_log(self, log_line):
        if "ERROR" in log_line:
            return {"status": "UNKNOWN_PATTERN", "method": "None", "raw_log": log_line}
        return {"status": "SUCCESS", "method": "Bi-LSTM-CRF", "confidence": 0.99, "structured_data": []}


def test_process_file_basic():
    parser = MockHybridParser()
    config = BatchConfig(batch_size=10, write_batch_size=5)
    pipeline = Pipeline(parser, config)

    # Create temp log file with 25 lines
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        for i in range(25):
            f.write(f"Log line {i}\n")
        input_path = f.name

    output_path = tempfile.mktemp(suffix='.json')

    try:
        stats = pipeline.process_file(input_path, output_path)
        assert stats["total_logs"] == 25
        assert stats["success_rate"] == 1.0
        assert stats["throughput_logs_per_sec"] > 0
        assert stats["total_time_sec"] >= 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read().strip()
        assert content.startswith("[")
        assert content.endswith("]")
        print("Basic pipeline processing: OK")
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_process_file_with_failures():
    parser = MockHybridParser()
    config = BatchConfig(batch_size=4, write_batch_size=5)
    pipeline = Pipeline(parser, config)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("INFO this is fine\n")
        f.write("ERROR this fails\n")
        f.write("DEBUG all good\n")
        f.write("ERROR another fail\n")
        input_path = f.name

    output_path = tempfile.mktemp(suffix='.json')

    try:
        stats = pipeline.process_file(input_path, output_path)
        assert stats["total_logs"] == 4
        assert stats["success_rate"] == 0.5
        print("Pipeline with failures: OK")
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_empty_file():
    parser = MockHybridParser()
    pipeline = Pipeline(parser)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        input_path = f.name

    output_path = tempfile.mktemp(suffix='.json')

    try:
        stats = pipeline.process_file(input_path, output_path)
        assert stats["total_logs"] == 0
        assert stats["success_rate"] == 0.0
        print("Empty file handling: OK")
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_blank_lines_skipped():
    parser = MockHybridParser()
    pipeline = Pipeline(parser, BatchConfig(batch_size=5, write_batch_size=5))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("line 1\n")
        f.write("\n")
        f.write("line 3\n")
        f.write("  \n")
        input_path = f.name

    output_path = tempfile.mktemp(suffix='.json')

    try:
        stats = pipeline.process_file(input_path, output_path)
        assert stats["total_logs"] == 2
        print("Blank lines skipped: OK")
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    test_process_file_basic()
    test_process_file_with_failures()
    test_empty_file()
    test_blank_lines_skipped()
    print("\nAll pipeline tests passed!")
