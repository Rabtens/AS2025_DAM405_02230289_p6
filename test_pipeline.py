"""
test_pipeline.py - automated checks of the Expected Results
-----------------------------------------------------------
Run AFTER train.py, compare_runs.py and register_model.py:
    python -m pytest -v test_pipeline.py
"""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from config import EXPERIMENT_NAME, MODEL_NAME, load_data, setup_mlflow

setup_mlflow()
client = MlflowClient()
Xtr, Xte, ytr, yte = load_data()


def test_experiment_has_multiple_comparable_runs():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    assert len(runs) >= 9
    for col in ("params.n_estimators", "params.max_depth", "metrics.accuracy"):
        assert runs[col].notna().all()          # every run logged the same fields


def test_every_run_logged_a_model_artifact():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    for run_id in runs["run_id"]:
        assert mlflow.sklearn.load_model(f"runs:/{run_id}/model") is not None


def test_model_registered_with_at_least_one_version():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) >= 1
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version


def test_registered_model_loads_and_predicts():
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    preds = model.predict(Xte[:5])
    assert preds.shape == (5,) and set(preds) <= {0, 1, 2}
    assert (model.predict(Xte) == yte).mean() >= 0.9
