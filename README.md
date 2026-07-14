# housing-price-ml

An end-to-end ML project: train a house-price model on a real public dataset, track experiments with MLflow, serve predictions through a FastAPI endpoint, and explore it all in an interactive Streamlit dashboard.

## Why this project

The California Housing dataset (Pace & Barry, 1997) is a real dataset derived from the 1990 US Census — median house values across California block groups, along with income, occupancy, and location features. It's small enough to train in seconds but real enough to demonstrate an actual ML workflow: feature prep, train/test split, experiment tracking, model serialization, and serving — the parts of "MLOps" that matter for a portfolio piece. It ships with scikit-learn (`fetch_california_housing`) and downloads directly from a public source, no Kaggle account or API key needed.

## Architecture

```
sklearn.datasets.fetch_california_housing()  (real public data, auto-downloaded)
        │
        ▼
   src/train.py
        │  trains a GradientBoostingRegressor, logs params/metrics/model to MLflow
        ▼
   mlruns/  (MLflow tracking store)          models/model.joblib (serialized model for serving)
                                                       │
                                              ┌────────┴────────┐
                                              ▼                 ▼
                                     app/main.py (FastAPI)   dashboard.py (Streamlit)
                                              │                 │
                                       POST /predict      interactive prediction form +
                                                           metrics + feature importance
```

**Stack:** scikit-learn, MLflow, FastAPI, Streamlit, Docker. All open source.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Train the model (downloads the dataset on first run, ~20k rows)
python src/train.py

# 2. Inspect experiment runs
mlflow ui   # opens http://localhost:5000

# 3. Serve predictions
uvicorn app.main:app --reload --port 8000
```

Then, in another terminal:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "MedInc": 8.3,
  "HouseAge": 41,
  "AveRooms": 6.98,
  "AveBedrms": 1.02,
  "Population": 322,
  "AveOccup": 2.55,
  "Latitude": 37.88,
  "Longitude": -122.23
}'
```

Response: `{"predicted_median_house_value_100k": 4.53}` (units match the original dataset: hundreds of thousands of USD).

Or with Docker:

```bash
docker build -t housing-price-ml .
docker run -p 8000:8000 housing-price-ml
```

(the Docker image trains the model at build time so it's ready to serve immediately)

## Dashboard

```bash
python src/train.py   # if you haven't already
streamlit run dashboard.py
```

Gives you an interactive form to try predictions against the trained model, plus the last
run's metrics (MAE/RMSE/R²), feature importances, and an actual-vs-predicted scatter plot
(the scatter plot needs the dataset to be reachable to rebuild the test set; the rest of the
dashboard works fully offline off the saved model).

## Project layout

```
housing-price-ml/
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data.py       # loads + splits the dataset
│   └── train.py      # trains model, logs to MLflow, saves models/model.joblib
├── app/
│   ├── main.py        # FastAPI app with /predict and /health
│   └── schemas.py     # pydantic request/response models
├── models/             # trained model artifact (generated, gitignored)
├── dashboard.py         # Streamlit dashboard
└── tests/
    ├── test_data.py
    ├── test_train.py
    └── test_app.py
```

## Tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

`test_app.py` trains a tiny model on a handful of rows so the API can be tested without needing the full training run first.

## Possible extensions

- Swap the file-based MLflow tracking store for a real MLflow Tracking Server + Postgres backend
- Register the best model in the MLflow Model Registry and load it into the API by stage (`Production`/`Staging`)
- Add drift/monitoring (Evidently AI) to compare live prediction inputs against training data distribution
- Add a `/predict/batch` endpoint for scoring a CSV of properties at once
