import pytest
import pandas as pd
import numpy as np

pytest.importorskip("xgboost", reason="xgboost not installed")

from python.primihub.local.ml_xgb.base import XGBoostModel


class TestXGBoostModel:
    def test_init_default(self):
        model = XGBoostModel()
        assert model is not None

    def test_train_and_predict(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
        })
        y = pd.Series((X["x1"] + X["x2"] > 0).astype(int))
        model = XGBoostModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_predict_proba(self):
        np.random.seed(42)
        X = pd.DataFrame({"x": np.random.randn(20)})
        y = pd.Series((X["x"] > 0).astype(int))
        model = XGBoostModel()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(y)

    def test_save_load(self, tmp_path):
        np.random.seed(42)
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 0, 1, 1, 1])
        model = XGBoostModel()
        model.fit(X, y)
        path = str(tmp_path / "model.json")
        model.save(path)
        loaded = XGBoostModel.load(path)
        preds_orig = model.predict(X)
        preds_loaded = loaded.predict(X)
        assert (preds_orig == preds_loaded).all()
