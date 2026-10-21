"""PSI must align positional model rows and preserve prediction IDs.

Run the real algorithm bodies with deterministic PSI/transport doubles, using
real pandas operations. Native KKRT integration is covered by cluster runs.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype, is_string_dtype


FL_ROOT = Path(__file__).resolve().parents[1]


def alignment(intersection):
    path = FL_ROOT / "psi/psi.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_alignment")
    executor = Mock()
    executor.run.return_value = intersection
    namespace = {
        "is_integer_dtype": is_integer_dtype, "is_string_dtype": is_string_dtype,
        "TwoPartyPsi": lambda: executor,
        "PsiType": SimpleNamespace(ECDH="ECDH", KKRT="KKRT"),
        "DataType": SimpleNamespace(Interger="integer", String="string"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["sample_alignment"], executor


@pytest.mark.parametrize("protocol", ["KKRT", "ECDH"])
def test_parties_get_same_order_and_clean_index(protocol):
    align, executor = alignment([9, 2, 5])
    host = pd.DataFrame({"id": [9, 2, 5, 100], "x": [90, 20, 50, 1000]}, index=[8, 4, 6, 2])
    guest = pd.DataFrame({"id": [5, 200, 9, 2], "z": [500, 20000, 900, 200]})
    roles = {"host": "Bob", "guest": ["Charlie"]}
    first = align(host, "id", roles, protocol)
    second = align(guest, "id", roles, protocol)
    assert first.id.tolist() == second.id.tolist() == [2, 5, 9]
    assert first.index.tolist() == second.index.tolist() == [0, 1, 2]
    assert first.x.tolist() == [20, 50, 90]
    assert second.z.tolist() == [200, 500, 900]
    assert roles == {"host": "Bob", "guest": ["Charlie"]}
    assert executor.run.call_args.kwargs["protocol"] == protocol


def test_string_identifiers_have_deterministic_order():
    align, _ = alignment(["z", "a", "m"])
    result = align(pd.DataFrame({"id": ["m", "z", "missing", "a"]}), "id",
                   {"host": "Bob", "guest": "Charlie"}, "KKRT")
    assert result.id.tolist() == ["a", "m", "z"]


@pytest.mark.parametrize("ids", [[1, 1], [1, None], ["a", None]])
def test_ambiguous_identifiers_rejected_before_native_psi(ids):
    align, executor = alignment([])
    with pytest.raises(ValueError, match="unique, non-null"):
        align(pd.DataFrame({"id": ids}), "id", {"host": "Bob", "guest": ["Charlie"]}, "KKRT")
    executor.run.assert_not_called()


def test_empty_intersection_has_no_stale_rows():
    align, _ = alignment([])
    frame = pd.DataFrame({"id": [1, 2], "x": [10, 20]}, index=[7, 9])
    result = align(frame, "id", {"host": "Bob", "guest": ["Charlie"]}, "KKRT")
    assert result.empty and result.index.tolist() == []
    assert list(result.columns) == ["id", "x"]
    assert frame.index.tolist() == [7, 9]


@pytest.mark.parametrize("psi", [True, False])
def test_predictions_keep_source_rows_and_ids(psi):
    path = FL_ROOT / "logistic_regression/vfl_host.py"
    parsed = ast.parse(path.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "LogisticRegressionHost")
    predict = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "predict")
    source = pd.DataFrame({"id": [9, 4, 7, 1], "x1": [.9, .4, .7, .1],
                           "y": [1, 0, 1, 0], "unused": ["a", "b", "c", "d"]}, index=[8, 4, 6, 2])
    align, _ = alignment([7, 1, 9])
    probabilities = np.array([.1, .7, .9] if psi else [.9, .4, .7, .1])
    model = SimpleNamespace(multiclass=False, compute_z=lambda x, _: x,
                            predict_prob=lambda _: probabilities)
    stored = {"selected_column": ["id", "x1", "y"], "id": "id", "label": "y",
              "transformer": SimpleNamespace(transform=lambda x: x), "model": model}
    save = Mock()
    namespace = {"pd": pd, "np": np, "sample_alignment": align,
        "MultiGrpcClients": lambda **kw: SimpleNamespace(recv_all=lambda _: []),
        "load_pickle_file": lambda _: stored, "read_data": lambda **kw: source.copy(),
        "check_array": lambda x, **kw: np.asarray(x, dtype=float), "save_csv_file": save}
    exec(compile(ast.Module(body=[predict], type_ignores=[]), str(path), "exec"), namespace)
    instance = SimpleNamespace(roles={"host": "Bob", "guest": ["Charlie"]},
        role_params={"self_name": "Bob", "others_role": "guest", "model_path": "model",
                     "data": {}, "predict_path": "result"}, common_params={"psi": "KKRT"} if psi else {},
        node_info={}, task_info={})
    namespace["predict"](instance)
    result = save.call_args.args[0]
    assert result.id.tolist() == ([1, 7, 9] if psi else [9, 4, 7, 1])
    assert result.pred_prob.tolist() == probabilities.tolist()
    assert result.pred_y.tolist() == [int(p > .5) for p in probabilities]
    assert not result.isna().any().any()
    assert result.unused.tolist() == (["d", "c", "a"] if psi else ["a", "b", "c", "d"])


def data_loader_class():
    path = FL_ROOT / "utils/dataset.py"
    parsed = ast.parse(path.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "DataLoader")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DataLoader"]


@pytest.mark.parametrize("seed", [0, 17])
def test_party_batches_stay_aligned_across_epochs_and_other_tasks(seed):
    loader = data_loader_class()
    data = np.arange(23)
    host = loader(data, label=data + 100, batch_size=7, seed=seed)
    guest = loader(data, batch_size=7, seed=seed)
    orders = []
    for _ in range(3):
        host_batches = list(host)
        np.random.random(100)
        list(loader(data, batch_size=5, seed=99))
        guest_batches = list(guest)
        for (x, y), z in zip(host_batches, guest_batches, strict=True):
            np.testing.assert_array_equal(x, z)
            np.testing.assert_array_equal(y, x + 100)
        order = np.concatenate(guest_batches)
        np.testing.assert_array_equal(np.sort(order), data)
        orders.append(order)
    assert not np.array_equal(orders[0], orders[1])


def test_loader_does_not_modify_global_random_state():
    before = np.random.get_state()
    list(data_loader_class()(np.arange(20), batch_size=6, seed=17))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_unshuffled_loader_keeps_input_order_and_partial_batch():
    batches = list(data_loader_class()(np.arange(10), batch_size=4, shuffle=False, seed=0))
    assert [len(batch) for batch in batches] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate(batches), np.arange(10))
