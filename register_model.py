"""
register_model.py - Task 5: register the best model in the Model Registry
-------------------------------------------------------------------------
Takes the best run chosen by compare_runs.py (best_run.txt) - or a run id
passed on the command line - and registers its logged model 'model' under the
registered-model name 'wine-classifier'.

Each call creates a new version (1, 2, 3, ...). After registering we:
  * add a description to the registered model and to the version,
  * tag the version with the metrics that justified choosing it,
  * point the alias 'champion' at it. Aliases are the MLflow 3 replacement for
    the older Staging/Production stages: consumers load
    models:/wine-classifier@champion and a rollout is just moving the alias.

Usage:  python register_model.py            (uses best_run.txt)
        python register_model.py <RUN_ID>
"""

import sys

import mlflow
from mlflow import MlflowClient

from config import MODEL_NAME, setup_mlflow

setup_mlflow()
client = MlflowClient()

run_id = sys.argv[1] if len(sys.argv) > 1 else open("best_run.txt").read().strip()
run = client.get_run(run_id)

# Register the model logged in that run (the guide's 'runs:/<RUN_ID>/model')
result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print("Registered version:", result.version)

# Document what was registered and why
client.update_registered_model(
    MODEL_NAME, description="RandomForest classifier for the sklearn Wine dataset "
                            "(DAM405 Practical 6).")
client.update_model_version(
    MODEL_NAME, result.version,
    description=f"Best of the hyper-parameter grid: run {run.info.run_name} "
                f"(accuracy={run.data.metrics['accuracy']:.4f}, "
                f"cv_accuracy={run.data.metrics['cv_accuracy']:.4f}).")
for key in ("accuracy", "cv_accuracy", "log_loss"):
    client.set_model_version_tag(MODEL_NAME, result.version, key,
                                 f"{run.data.metrics[key]:.4f}")
client.set_model_version_tag(MODEL_NAME, result.version, "source_run", run.info.run_name)

# Mark this version as the one consumers should use
client.set_registered_model_alias(MODEL_NAME, "champion", result.version)
print(f"Alias 'champion' -> {MODEL_NAME} v{result.version} (run {run.info.run_name})")
