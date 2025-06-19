# IteraTex Classic ML Quick-Start

This guide shows you **end-to-end** steps to run IteraTex _locally_ with any **classic machine-learning model** (scikit-learn, XGBoost-sklearn API, LightGBM-sklearn API, etc.).  You will:

1.  Spin up the full Docker stack (ZooKeeper, Kafka, ingestion, scheduler, API).
2.  Register & promote your own model.
3.  Query the live prediction endpoint.

> If you later want to serve deep-learning models (PyTorch / TensorFlow / ONNX) see `docs/deep_learning.md` (to be created).

---

## 1. Prerequisites

| Tool | Version |
|------|---------|
| Docker Desktop | 4.x+ (WSL2 backend on Windows) |
| Docker Compose V2 | comes with Docker Desktop |
| Python | 3.11 (for local scripts / training) |
| Make | optional – simplifies commands |

---

## 2. Clone & build

```bash
# Clone the repo (or your fork)
git clone https://github.com/<you>/iteratex.git
cd iteratex

# (Optional) create a virtualenv for local scripts
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

---

## 3. Start the IteraTex stack

```bash
# Build containers and start services in background
docker compose up --build -d

# Check health
 docker compose ps --format "table {{.Name}}\t{{.State}}\t{{.Health}}"
```
Expected healthy services:
```
zookeeper   running   healthy
kafka       running   healthy
ingestion   running   running
scheduler   running   running
api         running   healthy
```

### Ports
| Service | URL |
|---------|-----|
| Kafka (host) | `localhost:9092` |
| FastAPI | <http://localhost:8000/docs> |
| Prometheus metrics | <http://localhost:8000/metrics> |

> Tip: `docker compose logs -f ingestion` to watch streaming ingestion.

---

## 4. Feed data to Kafka (optional)
If you have a JSON stream, push messages to topic `raw-events`:
```bash
docker exec -it $(docker compose ps -q kafka) kafka-console-producer \
  --bootstrap-server kafka:9092 --topic raw-events
```
Paste JSON lines, then `Ctrl-D` to stop.

---

## 5. Register & promote **any classic ML model**
Assume you have a fitted pipeline saved as `pipeline_model.pkl` and a **feature list** in the same order you trained on.

Create `scripts/register_model.py` (or run inside a notebook):
```python
"""Register and promote a scikit-learn model into IteraTex."""
import time, shutil, pandas as pd
from iteratex.model_registry import utils as reg
from iteratex.model_registry.metadata import ModelMetadata

MODEL_SRC = r"C:\path\to\pipeline_model.pkl"
FEATURES = [  # ordered list
    "url", "length_url", "length_hostname", ...
]
PRIMARY_METRIC = {"accuracy": 0.93}

run_id = time.strftime("%Y%m%d-%H%M%S")
run_dir = reg.create_run_dir(run_id)
shutil.copy(MODEL_SRC, run_dir / "model.pkl")

meta = ModelMetadata(
    version=run_id,
    metrics=PRIMARY_METRIC,
    features=FEATURES,
    hyperparameters={},
    training_data_version="unknown",
)
(run_dir / "metadata.json").write_text(meta.model_dump_json(indent=2))

# --- smoke-test patch for DataFrame input (optional)
from pandas import DataFrame
import joblib, json
model = joblib.load(run_dir/"model.pkl")
sample = {f: ("test.com" if f=="url" else 0) for f in FEATURES}
model.predict(DataFrame([sample], columns=FEATURES))  # raises if incompatible
# ---

reg.promote(run_id)
print("Promoted", run_id)
```
Run it:
```bash
python scripts/register_model.py
```
If it prints `Promoted 20250617-084112` your model is now live.

---

## 6. Query the API

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "url": "http://foo.com", "length_url": 20, ... fill remaining ...
         }'
```
Example response:
```json
{"prediction": 1, "model_version": "20250617-084112"}
```

---

## 7. Retraining & CI/CD (optional)
- **Scheduler** automatically runs `scripts/retrain.py` every 30 min (see `RET_...` env vars).  Plug your own trainer/evaluator.
- Add tests under `tests/` and they will run in GitHub Actions (workflow already present).

---

## 8. Stopping services

```bash
docker compose down -v   # removes volumes
```

---

## 9. Troubleshooting
| Symptom | Fix |
|---------|-----|
| `docker compose up` exits instantly | Run `docker compose logs` to view error; ensure Docker Desktop/WSL2 is running. |
| Smoke test fails `Expected 2D array` | Provide DataFrame input (see patch above). |
| API returns `422 Unprocessable Entity` | JSON keys / types don’t match feature list. |
| High latency | Check `/metrics`; increase resources or optimize model. |

---

## 10. Next Steps
- Consult `docs/deep_learning.md` for PyTorch/TensorFlow/ONNX support.
- Add Prometheus & Grafana stack if you want dashboards.
- Configure alerting via `prometheus/alerts.yml`.

Enjoy production-grade self-training ML with IteraTex! 🎉
