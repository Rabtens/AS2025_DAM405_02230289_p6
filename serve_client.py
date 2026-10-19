"""
serve_client.py - Optional: call the registered model served over REST
-----------------------------------------------------------------------
First start the server in another terminal (from this folder):

    mlflow models serve -m "models:/wine-classifier/latest" -p 5001 --env-manager local

(--env-manager local reuses the current Python environment instead of building
a new one. The guide's command contains a typo, "wineclassifier"; the registered
name is "wine-classifier".)

This script then POSTs five test samples to /invocations and prints the reply.
"""

import requests

from config import load_data

Xtr, Xte, ytr, yte = load_data()

payload = {"inputs": Xte[:5].tolist()}            # MLflow scoring-server JSON format
r = requests.post("http://127.0.0.1:5001/invocations", json=payload, timeout=30)
r.raise_for_status()
print("HTTP", r.status_code, "->", r.json())
print("True labels:", yte[:5].tolist())
