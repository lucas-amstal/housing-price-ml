import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import FEATURE_NAMES, load_dataframe  # noqa: E402


def test_load_dataframe_shape_and_columns():
    X, y = load_dataframe()
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == len(y)
    assert len(X) > 1000  # California Housing has ~20,640 rows


def test_target_is_positive():
    _, y = load_dataframe()
    assert (y > 0).all()
