import time
from .batch_config import BatchConfig
from .stream_handler import BatchStreamer, JSONWriter

class Pipeline:
    """
    End-to-end pipeline:
    1. Load config
    2. Stream log batches
    3. Tokenize
    4. Chunker inference + confidence
    5. Route (chunker or siamese)
    6. Build structured JSON
    7. Write output
    8. Collect metrics
    """
    def __init__(self, hybrid_parser, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.parser = hybrid_parser
        
    def process_file(self, input_path: str, output_path: str):
        streamer = BatchStreamer(input_path, self.config.batch_size)
        writer = JSONWriter(output_path, self.config.write_batch_size)
        
        total_logs = 0
        success_count = 0
        start_time = time.time()
        
        for batch in streamer.get_batches():
            total_logs += len(batch)
            for log_line in batch:
                result = self.parser.parse_log(log_line)
                writer.add_record(result)
                if result.get("metadata", {}).get("status") in ["SUCCESS", "RESOLVED"]:
                    success_count += 1
                    
        writer.close()
        end_time = time.time()
        
        stats = {
            "total_logs": total_logs,
            "success_rate": success_count / max(1, total_logs),
            "throughput_logs_per_sec": total_logs / max(0.001, end_time - start_time),
            "total_time_sec": end_time - start_time,
        }
        return stats
