# DAM405 Practical 6 - Managing the ML Lifecycle with MLflow

Takes a RandomForest classifier for the scikit-learn **Wine** dataset through the full
MLflow lifecycle: **track → compare → register → load (→ serve)**.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (MLflow 3.x, scikit-learn, pandas, requests, pytest)

Check the install (Task 1):

```bash
mlflow --version
```

## How to run

Run every command from this folder so all scripts share the same tracking store.

```bash
python train.py              # Task 2 + 3 – 9 tracked runs in experiment "wine-classifier"
mlflow ui                    # Task 4 – open http://localhost:5000, sort runs by accuracy / cv_accuracy
python compare_runs.py       # Task 4 – same comparison in code -> runs_comparison.csv, best_run.txt
python register_model.py     # Task 5 – registers the best run as "wine-classifier" (+ alias "champion")
python load_and_predict.py   # Task 6 – loads models:/wine-classifier/latest and predicts
python -m pytest -v test_pipeline.py   # checks the Expected Results
```

Or run everything at once: `bash run_all.sh` (the UI still needs `mlflow ui`).

### Optional – serve the registered model over REST

```bash
mlflow models serve -m "models:/wine-classifier/latest" -p 5001 --env-manager local
python serve_client.py       # in a second terminal: POSTs 5 samples to /invocations
```

The guide's command spells the model `wineclassifier`; the registered name is `wine-classifier`.
On Windows PowerShell set the store first: `$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"`
(on Linux/macOS: `export MLFLOW_TRACKING_URI=sqlite:///mlflow.db`).

## Project structure

| File | Purpose | Task |
|---|---|---|
| `config.py` | Tracking URI, experiment/model names, data split shared by all scripts | – |
| `train.py` | `run(n_estimators, max_depth)` logs params, metrics and the model; 3×3 grid | 2, 3 |
| `compare_runs.py` | Searches runs, ranks them (accuracy → CV accuracy → log loss → fewer trees) | 4 |
| `register_model.py` | `mlflow.register_model('runs:/<RUN_ID>/model', 'wine-classifier')`, description, tags, alias | 5 |
| `load_and_predict.py` | Loads `models:/wine-classifier/latest` and `@champion`, predicts | 6 |
| `serve_client.py` | Calls the REST server started by `mlflow models serve` | optional |
| `test_pipeline.py` | 4 pytest checks of the Expected Results | – |
| `run_all.sh` | Runs the whole pipeline | – |

## Where MLflow stores things

MLflow 3.x keeps run metadata, metrics and the Model Registry in **`mlflow.db`** (SQLite) and the
model files in **`mlruns/`**. Both are created in this folder the first time `train.py` runs.

## Notes on MLflow 3.x

- `log_model(clf, name="model")` replaces the older `artifact_path="model"` argument.
- sklearn models are saved with **skops** instead of pickle. RandomForest trees must be trusted
  explicitly (`skops_trusted_types=["sklearn.tree._tree.Tree"]` in `train.py`); without it,
  logging fails with "untrusted types".
- Model *stages* (Staging/Production) are deprecated in favour of *aliases*; `register_model.py`
  sets the alias `champion`.

# AS2025_DAM405_02230289_p6
