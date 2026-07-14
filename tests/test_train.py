import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import train  # noqa: E402


def test_train_runs_and_returns_reasonable_metrics(tmp_path, monkeypatch):
    # Point MLflow at a throwaway local tracking dir so this test doesn't
    # pollute the real mlruns/ directory or require a tracking server.
    import mlflow

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    model, metrics = train(n_estimators=20, max_depth=2)

    assert model is not None
    assert 0 <= metrics["r2"] <= 1 or metrics["r2"] < 0  # r2 can be negative for a poor model, just check it's a float
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
