# IteraTex Phishing Detection - End-to-End Test

This directory contains scripts to test the end-to-end flow of the IteraTex phishing detection system.

## Prerequisites

1. Python 3.8+
2. Kafka (for a complete test)
3. Required Python packages (install with `pip install -r requirements-dev.txt`)

## Test Components

1. **Data Generation**: `generate_test_data.py`
   - Generates sample phishing and legitimate URL data
   - Saves data in both JSONL and CSV formats

2. **End-to-End Test**: `test_end_to_end.py`
   - Tests the complete pipeline:
     1. Data generation
     2. Data ingestion (simulated)
     3. Model training
     4. Model promotion
     5. API testing

## Running the Test

1. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run the end-to-end test**:
   ```bash
   python test_end_to_end.py
   ```

## Manual Testing

If you want to test components individually:

1. **Generate test data**:
   ```bash
   python generate_test_data.py --samples 1000 --output test_data/
   ```

2. **Start the API server**:
   ```bash
   uvicorn iteratex.serving.main:app --reload
   ```

3. **Test the API**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"feature1": 0.5, "feature2": 0.3}'
   ```

## Notes

- The test script simulates Kafka for simplicity. For a complete test, set up a local Kafka instance.
- The API server must be running for the prediction test to work.
- Model training uses the phishing dataset we prepared earlier.
