import pytest
from python.primihub.local.training_logger.base import (
    LogEntry,
    MetricsTracker,
)


class TestLogEntry:
    def test_create_entry(self):
        entry = LogEntry(
            timestamp="2024-01-01", level="INFO", message="test",
            epoch=1, metrics={"loss": 0.5})
        assert entry.epoch == 1
        assert entry.metrics["loss"] == 0.5

    def test_entry_to_dict(self):
        entry = LogEntry(
            timestamp="2024-01-01", level="INFO", message="test",
            epoch=2, metrics={"loss": 0.3})
        d = entry.to_dict()
        assert "epoch" in d


class TestMetricsTracker:
    def test_log_metrics(self):
        tracker = MetricsTracker()
        tracker.log_metrics({"loss": 0.5, "acc": 0.8}, epoch=1)
        tracker.log_metrics({"loss": 0.3, "acc": 0.9}, epoch=2)
        best = tracker.get_best_metrics()
        assert "loss" in best

    def test_best_loss(self):
        tracker = MetricsTracker()
        tracker.log_metrics({"loss": 0.5}, epoch=1)
        tracker.log_metrics({"loss": 0.3}, epoch=2)
        best = tracker.get_best_metrics()
        assert best["loss"]["value"] == 0.3

    def test_best_accuracy(self):
        tracker = MetricsTracker()
        tracker.log_metrics({"accuracy": 0.8}, epoch=1)
        tracker.log_metrics({"accuracy": 0.95}, epoch=2)
        best = tracker.get_best_metrics()
        assert best["accuracy"]["value"] == 0.95
