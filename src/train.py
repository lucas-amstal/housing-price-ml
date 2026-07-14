"""
Trains a house-price regression model on the California Housing dataset,
logs the run (params/metrics/model) to MLflow, and saves a serving-ready
copy of the model to models/model.joblib for the FastAPI app.

Usage:
    python src/train.py
    python src/train.py --n-estimators 300 --max-depth 4
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURE_NAMES, load_train_test_split  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def train(n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.1, random_state: int = 42):
    X_train, X_test, y_train, y_test = load_train_test_split(random_state=random_state)

    mlflow.set_experiment("california-housing-price")

    with mlflow.start_run():
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
        mlflow.log_params(params)

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": mean_squared_error(y_test, preds) ** 0.5,
            "r2": r2_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head(3))

        print(f"Run: {mlflow.active_run().info.run_id}")
        print(f"Metrics: {metrics}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        METADATA_PATH.write_text(
            json.dumps({"feature_names": FEATURE_NAMES, "params": params, "metrics": metrics}, indent=2)
        )
        print(f"Saved serving model to {MODEL_PATH}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    train(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
