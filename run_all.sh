#!/usr/bin/env bash
# Runs the whole practical end to end (Tasks 2, 3, 4-in-code, 5, 6 and tests).
# Start the UI separately with:  mlflow ui   (then open http://localhost:5000)
set -e
cd "$(dirname "$0")"
python train.py             # Tasks 2 + 3: 9 tracked runs
python compare_runs.py      # Task 4: leaderboard + best_run.txt
python register_model.py    # Task 5: register best run as wine-classifier
python load_and_predict.py  # Task 6: load from the registry and predict
python -m pytest -v test_pipeline.py
