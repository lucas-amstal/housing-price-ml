"""
Loads the California Housing dataset (Pace & Barry, 1997) — a real dataset
derived from the 1990 US Census, distributed with scikit-learn and downloaded
from a public source on first use, no API key required.

https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_NAME = "MedHouseVal"


def load_dataframe():
    """Returns (X, y) as pandas DataFrame/Series with readable column names."""
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data[FEATURE_NAMES]
    y = bunch.target
    return X, y


def load_train_test_split(test_size: float = 0.2, random_state: int = 42):
    X, y = load_dataframe()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
