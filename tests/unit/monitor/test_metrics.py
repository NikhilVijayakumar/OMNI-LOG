import time
from features.monitor.metrics import (
    calculate_parsing_accuracy,
    calculate_entity_f1,
    calculate_template_accuracy,
    PerformanceMonitor
)


def _approx(val, precision=0.001):
    return abs(val) < precision


def test_parsing_accuracy():
    assert calculate_parsing_accuracy(17, 20) == 0.85
    assert calculate_parsing_accuracy(0, 0) == 0.0
    assert calculate_parsing_accuracy(20, 20) == 1.0
    assert calculate_parsing_accuracy(0, 10) == 0.0
    print("Parsing accuracy: OK")


def test_entity_f1():
    assert calculate_entity_f1(1.0, 1.0) == 1.0
    assert calculate_entity_f1(0.5, 0.5) == 0.5
    assert calculate_entity_f1(0.0, 0.0) == 0.0
    expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
    assert _approx(calculate_entity_f1(0.8, 0.6) - expected)
    assert calculate_entity_f1(0.0, 1.0) == 0.0
    print("Entity F1: OK")


def test_template_accuracy():
    assert calculate_template_accuracy(45, 50) == 0.9
    assert calculate_template_accuracy(0, 0) == 0.0
    assert calculate_template_accuracy(50, 50) == 1.0
    print("Template accuracy: OK")


def test_performance_monitor():
    pm = PerformanceMonitor()
    assert pm.get_throughput() == 0
    assert pm.get_avg_latency() == 0

    pm.start_batch()
    time.sleep(0.01)
    pm.end_batch(batch_size=10)

    pm.start_batch()
    time.sleep(0.01)
    pm.end_batch(batch_size=20)

    assert pm.total_logs == 30
    assert pm.get_avg_latency() > 0
    assert pm.get_throughput() > 0
    print("PerformanceMonitor: OK")


if __name__ == "__main__":
    test_parsing_accuracy()
    test_entity_f1()
    test_template_accuracy()
    test_performance_monitor()
    print("\nAll metrics tests passed!")
