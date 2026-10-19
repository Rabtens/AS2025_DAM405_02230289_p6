"""
compare_runs.py - Task 4 (compare runs) done in code as well as in the UI
-------------------------------------------------------------------------
The MLflow UI (`mlflow ui`) is the main way to compare runs. This script does
the same comparison programmatically so the choice of "best run" is
repeatable and documented:

  1. search all runs in the experiment, sorted by accuracy (highest first)
  2. break ties in test accuracy by 5-fold CV accuracy (higher first), then
     test log loss (lower = more confident correct predictions), then
     n_estimators (smaller = cheaper to serve)
  3. print the leaderboard, save it to runs_comparison.csv, and write the
     best run id to best_run.txt for register_model.py
"""

import mlflow
import pandas as pd

from config import EXPERIMENT_NAME, setup_mlflow

setup_mlflow()


def leaderboard() -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                              filter_string="attributes.status = 'FINISHED'",
                              order_by=["metrics.accuracy DESC"])
    cols = {"run_id": "run_id", "tags.mlflow.runName": "run_name",
            "params.n_estimators": "n_estimators", "params.max_depth": "max_depth",
            "metrics.accuracy": "accuracy", "metrics.f1_macro": "f1_macro",
            "metrics.cv_accuracy": "cv_accuracy", "metrics.log_loss": "log_loss",
            "metrics.train_time_s": "train_time_s"}
    df = runs[list(cols)].rename(columns=cols)
    df["n_estimators"] = df["n_estimators"].astype(int)
    # tie-break: test accuracy, then CV accuracy, then log loss, then fewer trees
    df = df.sort_values(["accuracy", "cv_accuracy", "log_loss", "n_estimators"],
                        ascending=[False, False, True, True]).reset_index(drop=True)
    df.index += 1  # rank starting at 1
    return df


if __name__ == "__main__":
    board = leaderboard()
    pd.set_option("display.width", 140)
    print(board.drop(columns="run_id").round(4).to_string())

    board.to_csv("runs_comparison.csv", index_label="rank")
    best = board.iloc[0]
    with open("best_run.txt", "w") as f:
        f.write(best["run_id"])
    print(f"\nBest run: {best['run_name']}  run_id={best['run_id']}  "
          f"accuracy={best['accuracy']:.4f}  cv_accuracy={best['cv_accuracy']:.4f}  "
          f"log_loss={best['log_loss']:.4f}")
    print(f"{int((board['accuracy'] == best['accuracy']).sum())} run(s) share the top accuracy")
