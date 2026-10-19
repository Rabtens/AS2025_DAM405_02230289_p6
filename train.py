"""
train.py - Task 2 (track a training run) and Task 3 (run several experiments)
-----------------------------------------------------------------------------
run(n_estimators, max_depth) wraps one training in an MLflow run and logs:
  * parameters : n_estimators, max_depth (+ random_state, test_size)
  * metrics    : accuracy (required by the guide), plus macro F1, test
                 log loss, 5-fold CV accuracy (+ std) and training time,
                 so that ties in test accuracy can be broken
  * artifact   : the fitted model under the name 'model', with a signature
                 and input example so it can later be served over REST
  * tags       : dataset and model family, to make runs searchable

The __main__ block runs the 3 x 3 hyper-parameter grid from Task 3,
giving 9 comparable runs in the 'wine-classifier' experiment.
"""

import time

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_val_score

from config import RANDOM_STATE, load_data, setup_mlflow

setup_mlflow()
Xtr, Xte, ytr, yte = load_data()


def run(n_estimators, max_depth):
    """Train one RandomForest inside an MLflow run and log everything about it."""
    run_name = f"rf_n{n_estimators}_d{max_depth}"
    with mlflow.start_run(run_name=run_name):
        # --- train -------------------------------------------------------------
        t0 = time.perf_counter()
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=RANDOM_STATE).fit(Xtr, ytr)
        train_s = time.perf_counter() - t0

        # --- evaluate on the held-out test set -----------------------------------
        y_pred = clf.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        f1 = f1_score(yte, y_pred, average="macro")
        # log loss uses predicted probabilities, so it still separates models
        # that all get every test label right (confidence matters)
        ll = log_loss(yte, clf.predict_proba(Xte))
        # 5-fold cross-validation on the training set: a steadier estimate
        # than a 36-sample test set, used as the main tie-breaker
        cv = cross_val_score(RandomForestClassifier(n_estimators=n_estimators,
                                                    max_depth=max_depth,
                                                    random_state=RANDOM_STATE),
                             Xtr, ytr, cv=5, scoring="accuracy")

        # --- log parameters (the inputs that define the run) ---------------------
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)        # None is logged as "None"
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)

        # --- log metrics (the outcomes we compare runs on) -----------------------
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("log_loss", ll)
        mlflow.log_metric("cv_accuracy", cv.mean())
        mlflow.log_metric("cv_accuracy_std", cv.std())
        mlflow.log_metric("train_time_s", train_s)

        # --- tags make runs easy to filter in the UI -----------------------------
        mlflow.set_tags({"dataset": "sklearn-wine", "model_family": "RandomForest"})

        # --- log the model artifact ('model' is the artifact name used later by
        #     runs:/<RUN_ID>/model when registering) -------------------------------
        #     MLflow 3.x saves sklearn models with skops (safer than pickle) and
        #     refuses types it cannot verify. The tree structure inside a
        #     RandomForest is one of them; we built this model ourselves, so we
        #     explicitly trust exactly that one type and nothing else.
        signature = infer_signature(Xtr, clf.predict(Xtr))
        mlflow.sklearn.log_model(clf, name="model", signature=signature,
                                 input_example=Xtr[:3],
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"n={n_estimators} depth={max_depth} acc={acc:.4f} "
              f"cv_acc={cv.mean():.4f} log_loss={ll:.4f}")


if __name__ == "__main__":
    # Task 3: 3 x 3 grid -> 9 runs to compare
    for n in (50, 100, 200):
        for d in (3, 5, None):
            run(n, d)
