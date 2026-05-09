# src/features/monitor/metrics.py
import time

def calculate_parsing_accuracy(correct_parsed: int, total_logs: int) -> float:
    return correct_parsed / max(1, total_logs)

def calculate_entity_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_template_accuracy(correct_matches: int, total_preds: int) -> float:
    return correct_matches / max(1, total_preds)

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.total_logs = 0
        self.latency_per_batch = []
        
    def start_batch(self):
        self.batch_start = time.time()
        
    def end_batch(self, batch_size: int):
        t = time.time() - self.batch_start
        self.latency_per_batch.append(t)
        self.total_logs += batch_size
        
        if self.start_time is None:
            self.start_time = time.time() - t
            
    def get_throughput(self) -> float:
        if not self.start_time:
            return 0
        elapsed = time.time() - self.start_time
        return self.total_logs / max(0.001, elapsed)
        
    def get_avg_latency(self) -> float:
        if not self.latency_per_batch:
            return 0
        return sum(self.latency_per_batch) / len(self.latency_per_batch) * 1000  # ms/batch
