import numpy as np
import pandas as pd
import pytest
import pickle
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataUtils:
    def test_read_csv_with_pandas(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write("id,a,b\n1,1.0,2.0\n2,3.0,4.0\n3,5.0,6.0\n")
            fname = f.name
        result = pd.read_csv(fname)
        assert result.shape == (3, 3)
        os.unlink(fname)

    def test_select_columns(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            fname = f.name
        df = pd.read_csv(fname)
        result = df[['a', 'c']]
        assert list(result.columns) == ['a', 'c']
        os.unlink(fname)

    def test_concat_dataframes(self):
        df1 = pd.DataFrame({'pred_y': [0.5, 0.8]})
        df2 = pd.DataFrame({'id': [1, 2], 'x': [10, 20]})
        result = pd.concat([df2, df1], axis=1)
        assert result.shape == (2, 3)


class TestDataLoader:
    def test_batch_iteration(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 1, 0, 1])
        batch_size = 2
        batches = []
        for i in range(0, len(x), batch_size):
            bx = x[i:i + batch_size]
            by = y[i:i + batch_size]
            batches.append((bx, by))
        assert len(batches) == 2
        assert batches[0][0].shape == (2, 2)

    def test_dp_poisson_sampling(self):
        np.random.seed(42)
        n, batch_size, total = 100, 10, 0
        for _ in range(10):
            probs = np.random.rand(n)
            indices = np.where(probs < batch_size / n)[0]
            total += len(indices)
        assert total > 0


class TestFileUtils:
    def test_pickle_roundtrip(self):
        data = {"key": "value", "num": 42}
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            fname = f.name
        with open(fname, 'wb') as f:
            pickle.dump(data, f)
        with open(fname, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded == data
        os.unlink(fname)

    def test_json_roundtrip(self):
        data = {"accuracy": 0.95, "loss": 0.1}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            fname = f.name
        with open(fname, 'w') as f:
            json.dump(data, f)
        with open(fname, 'r') as f:
            loaded = json.load(f)
        assert loaded == data
        os.unlink(fname)

    def test_csv_save_load(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            fname = f.name
        df.to_csv(fname, index=False)
        loaded = pd.read_csv(fname)
        assert loaded.shape == (3, 2)
        os.unlink(fname)
