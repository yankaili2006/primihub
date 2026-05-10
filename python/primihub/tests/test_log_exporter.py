import pytest
import json
import csv
import tempfile
import os
from python.primihub.local.log_exporter.base import (
    JSONExporter,
    CSVExporter,
)


class TestJSONExporter:
    def test_export_data(self, tmp_path):
        data = [
            {"epoch": 1, "loss": 0.5, "acc": 0.8},
            {"epoch": 2, "loss": 0.3, "acc": 0.9},
        ]
        path = str(tmp_path / "output.json")
        exporter = JSONExporter()
        exporter.export(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["epoch"] == 1

    def test_export_empty(self, tmp_path):
        path = str(tmp_path / "empty.json")
        exporter = JSONExporter()
        exporter.export([], path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == []


class TestCSVExporter:
    def test_export_session(self, tmp_path):
        data = {
            "session_id": "test_001",
            "logs": [
                {"step": 1, "loss": 0.5},
                {"step": 2, "loss": 0.3},
            ],
        }
        exporter = CSVExporter(export_metrics=False)
        output_dir = exporter.export(data, str(tmp_path / "export"))
        expected = os.path.join(output_dir, "export_logs.csv")
        assert os.path.exists(expected)
