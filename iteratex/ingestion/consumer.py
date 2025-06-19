"""Kafka consumer for IteraTex.

Changes in v1.1 (production-ready ingestion):
1. **Hourly partitioned parquet** – records are first buffered in memory and then flushed to
   ``data/training/YYYY-MM-DD/HH.parquet`` so no full-file rewrites occur.
2. **Micro-batching** – configurable batch size and/or flush interval to reduce I/O.
3. **Back-pressure** – ingest pauses when disk free space drops below a threshold.

Run with
    python -m iteratex.ingestion.consumer
Environment variables (optional)
    KAFKA_BOOTSTRAP_SERVERS
    KAFKA_TOPIC
    BATCH_SIZE                – default 500
    FLUSH_SECONDS             – default 60
    DISK_FREE_MIN_GB          – default 1
    METRICS_PORT              – default 8001
"""
import json
import os
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from loguru import logger
from prometheus_client import Counter, Gauge, start_http_server

# Prometheus metrics
MESSAGES_CONSUMED_TOTAL = Counter(
    "iteratex_messages_consumed_total",
    "Total messages consumed from Kafka",)
FLUSHES_TOTAL = Counter(
    "iteratex_flushes_total",
    "Number of parquet batch flushes to disk",)
MESSAGES_QUARANTINED_TOTAL = Counter(
    "iteratex_messages_quarantined_total",
    "Invalid records sent to quarantine topic",
)
BACKPRESSURE_ACTIVE = Gauge(
    "iteratex_backpressure", "1 when ingestion is paused due to back-pressure, else 0")


def _partitioned_path(ts: datetime) -> Path:
    """Return parquet path for given timestamp (UTC)."""
    root = Path(__file__).resolve().parents[2]
    base = root / "data" / "training" / ts.strftime("%Y-%m-%d")
    base.mkdir(parents=True, exist_ok=True)
    fname = f"{ts.strftime('%H')}.parquet"  # one file per hour
    return base / fname


from iteratex.preprocessing import validate_json

def _flush_buffer(buffer: List[Dict[str, Any]]):
    if not buffer:
        return
    ts = datetime.utcnow().replace(tzinfo=timezone.utc)
    path = _partitioned_path(ts)
    df = pd.DataFrame(buffer)
    # append=True requires pyarrow engine (pandas>=2.0)
    df.to_parquet(path, index=False, compression="snappy", append=True, engine="pyarrow")
    FLUSHES_TOTAL.inc()
    logger.info("Flushed %d records to %s", len(buffer), path)


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / 1_073_741_824  # bytes → GiB


def consume():
    """Consume Kafka messages, validate, buffer, and flush to hourly parquet files.
    Invalid messages are sent to a quarantine topic. Back-pressure activates when
    disk free space falls below `DISK_FREE_MIN_GB`.
    """
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    batch_size = int(os.getenv("BATCH_SIZE", "500"))
    flush_seconds = int(os.getenv("FLUSH_SECONDS", "60"))
    min_free_gb = int(os.getenv("DISK_FREE_MIN_GB", "1"))
    topic = os.getenv("KAFKA_TOPIC", "iteratex-data")
    quarantine_topic = os.getenv("QUARANTINE_TOPIC", "iteratex-quarantine")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="iteratex-consumer",
    )

    # Single producer instance is cheap and avoids per-message instantiation
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
    )

    logger.info("Started consumer on topic '%s'", topic)

    buffer: List[Dict[str, Any]] = []
    last_flush = time.time()
    data_dir = Path(__file__).resolve().parents[2] / "data"

    for msg in consumer:
        # Back-pressure check
        if _disk_free_gb(data_dir) < min_free_gb:
            BACKPRESSURE_ACTIVE.set(1)
            logger.warning("Low disk space (<%d GiB). Pausing ingestion…", min_free_gb)
            time.sleep(5)
            continue
        BACKPRESSURE_ACTIVE.set(0)

        raw = msg.value
        try:
            record = validate_json(raw)
            buffer.append(record)
            MESSAGES_CONSUMED_TOTAL.inc()
        except Exception as exc:
            # Send bad record to quarantine topic
            producer.send(quarantine_topic, raw)
            producer.flush(0)
            MESSAGES_QUARANTINED_TOTAL.inc()
            logger.warning("Record sent to quarantine due to validation error: %s", exc)
            continue

        now = time.time()
        if len(buffer) >= batch_size or (now - last_flush) >= flush_seconds:
            _flush_buffer(buffer)
            buffer.clear()
            last_flush = now





    
        # Back-pressure check
        if _disk_free_gb(data_dir) < min_free_gb:
            BACKPRESSURE_ACTIVE.set(1)
            logger.warning("Low disk space (<%d GiB). Pausing ingestion…", min_free_gb)
            time.sleep(5)
            continue
        BACKPRESSURE_ACTIVE.set(0)

        raw = msg.value
        try:
            record = validate_json(raw)
            buffer.append(record)
            MESSAGES_CONSUMED_TOTAL.inc()
        except Exception as exc:
            # Send bad record to quarantine topic
            producer.send(quarantine_topic, raw)
            producer.flush(0)
            MESSAGES_QUARANTINED_TOTAL.inc()
            logger.warning("Record sent to quarantine due to validation error: %s", exc)

        now = time.time()
        if len(buffer) >= batch_size or (now - last_flush) >= flush_seconds:
            _flush_buffer(buffer)
            buffer.clear()
            last_flush = now
            
def main():
    # Start Prometheus metrics server
    metrics_port = int(os.getenv("METRICS_PORT", "8001"))
    start_http_server(metrics_port)
    consume()


if __name__ == "__main__":
    main()
