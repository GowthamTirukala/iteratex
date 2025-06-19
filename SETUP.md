# IteraTex - Setup Guide with Docker

This guide will help you set up and run the IteraTex phishing detection system using Docker.

## Prerequisites

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Windows/macOS) or Docker Engine (for Linux)
2. Install [Git](https://git-scm.com/downloads)
3. (Optional) Install [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd iteratex
```

### 2. Build and Start Containers

Start all services (Kafka, Zookeeper, Ingestion, API):

```bash
docker-compose up -d
```

This will start the following services:
- Zookeeper (port 2181)
- Kafka (ports 9092, 29092)
- Ingestion service
- API server (port 8000)

### 3. Verify Services

Check if all containers are running:

```bash
docker-compose ps
```

### 4. Create Kafka Topic

Create a topic for phishing URLs:

```bash
docker-compose exec kafka kafka-topics --create --topic phishing-urls --partitions 1 --replication-factor 1 --bootstrap-server localhost:9092
```

### 5. Test the System

#### Send Test Data

```bash
# Generate test data
python scripts/generate_test_data.py --samples 10

# Send data to Kafka (using kcat)
docker run --network=host --rm confluentinc/cp-kafkacat kafkacat -b localhost:29092 -t phishing-urls -P -l test_data/phishing_test_data.jsonl
```

#### Train a Model

```bash
docker-compose exec scheduler python -m iteratex.training.phishing_trainer
```

#### Test the API

```bash
# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url_length":120,"num_dots":5,"num_hyphens":3,"num_underscore":2,"num_slash":8,"num_question":1,"num_equal":2,"num_percent":1,"num_digits":15,"has_ip":0,"has_at":1,"is_https":0,"domain_length":35,"num_subdomains":4}'
```

### 6. View Metrics

Access Prometheus metrics at:
```
http://localhost:8000/metrics
```

## Development Workflow

1. **Edit code** in your local repository
2. **Rebuild** the container when you change dependencies:
   ```bash
   docker-compose build
   docker-compose up -d
   ```
3. **View logs** for a service:
   ```bash
   docker-compose logs -f <service_name>
   ```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure ports 2181, 9092, 29092, and 8000 are available
2. **Kafka not starting**: Check Zookeeper logs first
3. **Docker out of memory**: Increase Docker's memory allocation in Docker Desktop settings

### Viewing Logs

```bash
# View all logs
docker-compose logs

# Follow logs for a specific service
docker-compose logs -f kafka
```

## Stopping the Services

```bash
docker-compose down
```

To remove all data (including Kafka topics and model registry):

```bash
docker-compose down -v
```
