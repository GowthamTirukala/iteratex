"""Generate test data for the phishing model."""

import json
import random
from pathlib import Path

import pandas as pd


def generate_sample_record(is_phishing: bool) -> dict:
    """Generate a single test record with realistic phishing features."""
    if is_phishing:
        return {
            "url_length": random.randint(50, 200),
            "num_dots": random.randint(3, 10),
            "num_hyphens": random.randint(2, 8),
            "num_underscore": random.randint(0, 5),
            "num_slash": random.randint(3, 15),
            "num_question": random.randint(0, 3),
            "num_equal": random.randint(0, 5),
            "num_percent": random.randint(0, 5),
            "num_digits": random.randint(5, 30),
            "has_ip": random.choice([0, 1]),
            "has_at": random.choice([0, 1]),
            "is_https": random.choice([0, 1]),
            "domain_length": random.randint(10, 50),
            "num_subdomains": random.randint(1, 5),
            "label": 1 if is_phishing else 0,
        }
    else:
        return {
            "url_length": random.randint(20, 100),
            "num_dots": random.randint(1, 3),
            "num_hyphens": random.randint(0, 2),
            "num_underscore": random.randint(0, 1),
            "num_slash": random.randint(1, 5),
            "num_question": random.randint(0, 1),
            "num_equal": random.randint(0, 2),
            "num_percent": random.randint(0, 1),
            "num_digits": random.randint(0, 10),
            "has_ip": 0,
            "has_at": 0,
            "is_https": 1,
            "domain_length": random.randint(5, 20),
            "num_subdomains": random.randint(1, 2),
            "label": 0,
        }


def generate_test_data(num_samples: int = 100, output_dir: Path = Path("test_data")):
    """Generate test data and save to files."""
    output_dir.mkdir(exist_ok=True)

    # Generate records
    records = []
    for i in range(num_samples):
        is_phishing = i % 2 == 0  # Alternate between phishing and legitimate
        records.append(generate_sample_record(is_phishing))

    # Save as JSONL
    jsonl_path = output_dir / "phishing_test_data.jsonl"
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Save as CSV for reference
    df = pd.DataFrame(records)
    csv_path = output_dir / "phishing_test_data.csv"
    df.to_csv(csv_path, index=False)

    print(f"Generated {num_samples} test records")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    generate_test_data(num_samples=100)
