"""Utility to publish N test messages to iteratex-data kafka topic."""
import json
import random
import sys
from kafka import KafkaProducer

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    bootstrap = sys.argv[2] if len(sys.argv) > 2 else "localhost:9092"
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
    )
    for i in range(n):
        msg = {
            "feature1": random.random(),
            "feature2": random.random(),
            "label": i % 2,
        }
        producer.send("iteratex-data", msg)
    producer.flush()
    print(f"Sent {n} test messages to iteratex-data on {bootstrap}")

if __name__ == "__main__":
    main()
