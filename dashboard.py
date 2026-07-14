"""
Streamlit dashboard for the California Housing price model.

Two things live here:
  1. An interactive prediction form (loads models/model.joblib directly, no
     need to run the FastAPI server separately)
  2. Model performance/explainability: metrics + feature importances from
     the last training run (models/model_metadata.json), plus an actual-vs-
     predicted scatter plot on a held-out test set if the dataset can be
     (re)loaded.

Run with:
    streamlit run dashboard.py

If models/model.joblib doesn't exist yet (e.g. a fresh clone on Streamlit
Community Cloud, where `python src/train.py` never runs), this dashboard
trains it automatically on first load.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="California Housing Price Model", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        with st.spinner("No trained model found yet — training one now (first run only, ~10s)..."):
            try:
                from train import train

                train()
            except Exception as exc:
                st.error(
                    f"Automatic training failed: {exc}\n\n"
                    "This usually means the dataset couldn't be downloaded "
                    "(needs internet access from wherever this app is running)."
                )
                return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    return json.loads(METADATA_PATH.read_text())


@st.cache_data
def try_load_test_set():
    """Best-effort: reload the dataset for an actual-vs-predicted plot.
    Returns None if the dataset can't be fetched (e.g. no internet)."""
    try:
        from data import load_train_test_split

        _, X_test, _, y_test = load_train_test_split()
        return X_test, y_test
    except Exception:
        return None


def prediction_form(model, feature_names):
    st.subheader("Try a prediction")
    defaults = {
        "MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
        "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23,
    }
    cols = st.columns(4)
    values = {}
    for i, name in enumerate(feature_names):
        with cols[i % 4]:
            values[name] = st.number_input(name, value=defaults.get(name, 0.0))

    if st.button("Predict median house value", type="primary"):
        row = pd.DataFrame([values])
        prediction = model.predict(row)[0]
        st.success(f"Predicted median house value: **${prediction * 100_000:,.0f}**")


def main():
    st.title("California Housing Price Model")
    st.caption("scikit-learn GradientBoostingRegressor, tracked with MLflow")

    model = load_model()
    if model is None:
        st.warning("Couldn't load or train a model — see the error above.")
        return

    metadata = load_metadata()
    feature_names = metadata["feature_names"] if metadata else list(model.feature_names_in_)

    prediction_form(model, feature_names)

    st.divider()

    if metadata:
        st.subheader("Last training run")
        m = metadata["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{m['mae']:.3f}")
        col2.metric("RMSE", f"{m['rmse']:.3f}")
        col3.metric("R²", f"{m['r2']:.3f}")
        st.caption(f"Params: {metadata['params']}")

    left, right = st.columns(2)

    with left:
        st.subheader("Feature importance")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
        st.bar_chart(importances)

    with right:
        st.subheader("Actual vs. predicted (test set)")
        test_data = try_load_test_set()
        if test_data is None:
            st.info(
                "Couldn't (re)load the California Housing dataset to build this chart "
                "(needs internet access). The prediction form and feature importances "
                "above still work off the trained model."
            )
        else:
            X_test, y_test = test_data
            preds = model.predict(X_test)
            chart_df = pd.DataFrame({"actual": y_test.values, "predicted": preds})
            st.scatter_chart(chart_df, x="actual", y="predicted")


if __name__ == "__main__":
    main()
