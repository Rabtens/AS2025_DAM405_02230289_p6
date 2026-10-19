"""
config.py - settings and data shared by every script in this practical
----------------------------------------------------------------------
Keeping the experiment name, model name and train/test split in one place
guarantees that training, comparison, registration and loading all refer to
exactly the same experiment, registered model and test data.

Tracking store (MLflow 3.x default, made explicit here):
  * run metadata (params, metrics, tags, registry) -> mlflow.db (SQLite)
  * artifacts (the pickled models, plots)          -> ./mlruns/
`mlflow ui` started in this folder reads the same store.
"""

from pathlib import Path

import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "wine-classifier"
MODEL_NAME = "wine-classifier"
RANDOM_STATE = 42


def setup_mlflow() -> None:
    """Point MLflow at the project's tracking store and select the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    """Wine dataset, 80/20 split with a fixed seed (as in the practical guide)."""
    X, y = load_wine(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
