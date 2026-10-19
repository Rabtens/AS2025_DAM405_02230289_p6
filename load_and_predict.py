"""
load_and_predict.py - Task 6: load the registered model and predict
-------------------------------------------------------------------
Loads the model from the Model Registry - not from a file path or a run id -
which is how a downstream service would consume it:

  models:/wine-classifier/latest    -> newest version (as in the guide)
  models:/wine-classifier@champion  -> whichever version the alias points to

Predicts the first five test samples, compares them with the true labels and
checks both URIs give identical predictions.
"""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score

from config import MODEL_NAME, load_data, setup_mlflow

setup_mlflow()
Xtr, Xte, ytr, yte = load_data()

# Which version does 'latest' resolve to?
latest = max(MlflowClient().search_model_versions(f"name='{MODEL_NAME}'"),
             key=lambda v: int(v.version))
print(f"Registry: {MODEL_NAME} latest version = v{latest.version} (run {latest.run_id[:8]})")

# --- the guide's Task 6 code -----------------------------------------------------
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
print("Loaded model prediction:", model.predict(Xte[:5]))
print("True labels            :", yte[:5])

# --- extra checks -----------------------------------------------------------------
print(f"Accuracy of loaded model on the full test set: "
      f"{accuracy_score(yte, model.predict(Xte)):.4f}")

champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
same = (champion.predict(Xte) == model.predict(Xte)).all()
print(f"models:/{MODEL_NAME}@champion gives identical predictions: {same}")
