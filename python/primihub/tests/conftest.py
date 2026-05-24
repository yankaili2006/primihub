"""Conftest for primihub/tests/ - ensures clean module state."""
import sys
import pytest


@pytest.fixture(scope="session", autouse=True)
def clean_mock_modules():
    """Clean up any mock modules left by other test directories.
    Runs before any test in this directory.
    
    Removes modules that may have been set up as mock namespace packages
    by other test files (e.g. test_fl_feature_engineering.py mocks
    primihub.FL.*, google.*, etc.)."""
    keys_to_clean = []
    for key in list(sys.modules.keys()):
        if (key.startswith("primihub.") or
                key.startswith("google") or
                key.startswith("grpc.") or
                key.startswith("ph_") or
                key.startswith("opt_")):
            keys_to_clean.append(key)
    for key in keys_to_clean:
        sys.modules.pop(key, None)
    # Remove top-level packages if they are stubs (no __file__)
    for top in ["primihub", "google"]:
        if top in sys.modules:
            m = sys.modules[top]
            if not getattr(m, "__file__", None):
                sys.modules.pop(top, None)
    yield
