# IteraTex

IteraTex is a self-training AI / ML infrastructure prototype that ingests real-time data, periodically retrains multiple model types, and automatically promotes better-performing models without downtime.

## Repository layout

```
iteratex/
│  pyproject.toml        – project & dependency definition (PEP 621)
│  README.md             – this file
│
├─ iteratex/             – Python package containing all runtime code
│   ├─ ingestion/        – Kafka consumers & data writers
│   ├─ serving/          – FastAPI inference application
│   ├─ training/         – trainers for various model types (TBD)
│   ├─ evaluation/       – evaluators & promotion logic (TBD)
│   ├─ registry/         – helpers for reading / writing model artefacts
│   └─ config/           – default YAML configs
│
└─ data/
    └─ training/         – streamed raw data is appended here (Parquet)
```

## Quick-start (local)

1. Install deps (requires Python 3.9+):

```bash
pip install -e .[dev]
```

2. Start the Kafka consumer:

```bash
python -m iteratex.ingestion.consumer
```

3. Run the FastAPI server:

```bash
uvicorn iteratex.serving.main:app --reload --host 0.0.0.0 --port 8000
```

4. Query the API:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature1": 1, "feature2": 2}'
```

### Notes
* The consumer appends each Kafka message to `data/training/training_data.parquet`.
* The serving app currently uses a **dummy model** that echoes a constant value; this will be replaced once trainers and evaluators are wired.
* All services expose Prometheus metrics at `/metrics`.
