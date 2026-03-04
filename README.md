# Electricity Theft Detection via Hybrid SLM + Time-Series Anomaly Detection

[![CI](https://github.com/amiraghabiglou/electricity-theft-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/amiraghabiglou/electricity-theft-detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready ML system designed to detect anomalous electricity consumption patterns indicative of energy theft. This project bridges the gap between robust statistical time-series analysis and modern Small Language Models (SLMs) to provide highly accurate, interpretable fraud investigation reports.
## 🏗️ Architecture

This system addresses the **critical weaknesses** of naive "text-as-input" approaches by using a **hybrid architecture**:

| Component | Technology | Role |
|-----------|-----------|------|
| **Feature Extraction** | TSFRESH | Extracts 700+ statistical features (trend, seasonality, entropy) |
| **Anomaly Detection** | Isolation Forest | Unsupervised scoring for novel theft patterns |
| **Classification** | XGBoost | High-precision fraud detection with IF features as input |
| **Reasoning Engine** | Phi-3/Llama-3-8B (4-bit GGUF) | Natural language investigation reports from structured outputs |

**Why this architecture?**
- ❌ **Avoids**: Token inflation from "Day 1: 12kWh..." textification (inefficient, lossy)
- ✅ **Uses**: Structured feature vectors → Statistical detection → LLM interpretation
- 🎯 **Result**: 10x faster inference, higher accuracy, interpretable outputs

## 📁 Repository Structure
```text
├── .github/workflows/
│   └── ci.yml               # CI/CD Pipeline (Test, Train, Drift, Build)
├── config/
│   └── pipeline_config.yaml  # Model & Pipeline configurations
├── docker/
│   ├── Dockerfile.api        # FastAPI service container
│   └── Dockerfile.training   # Training & Quantization environment
├── scripts/
│   ├── download_data.py      # SGCC dataset ingestion
│   ├── quantize_llm.py       # SLM (Phi-3) GGUF quantization
│   └── train_models.py       # Hybrid model training orchestrator
├── src/
│   ├── api/                  # FastAPI Application
│   ├── features/             # TSFRESH & Domain feature engineering
│   ├── llm/                  # SLM Reasoning & Report generation
│   ├── models/               # Hybrid Ensemble (Isolation Forest + XGBoost)
│   ├── monitoring/           # Data drift detection logic
│   ├── pipeline/             # End-to-end data processing
│   ├── schemas/              # Pydantic models & Feature mappings
│   └── workers/              # Celery task definitions
├── tests/
│   ├── integration/          # Pipeline & Resource isolation tests
│   └── unit/                 # Reasoning & Logic tests
├── docker-compose.yml        # Local development stack (App + Redis + Workers)
├── pyproject.toml            # Poetry dependency management
└── README.md
```

## 🚀 Quick Start (Docker / Reviewer Path)
The fastest way to evaluate this system is via Docker. No local Python environment is required.

### Prerequisites
- Docker and Docker Compose.

- 8GB+ RAM.

- Environment Variables: Zero-config required for default execution. The stack runs out-of-the-box using the defaults in docker-compose.yml.

### Installation

```bash
# 1. Clone the repository
git clone git@github.com:amiraghabiglou/electricity-theft-detection.git
cd electricity-theft-detection

# 2. Start the full stack (FastAPI, Redis, Celery Workers)
docker-compose up -d --build

# 3. Verify the API is running
curl http://localhost:8000/health
```
### 💻 Local Development (Developer Path)
For MLOps engineers and contributors, this project strictly manages dependencies via [Poetry](https://python-poetry.org/).

#### Hardware Requirements
- Minimum (Inference): 8GB RAM, 4 CPU cores, 10GB free disk space. Sufficient for running inference with Q4_K_M quantized models.

- Recommended (Training): 16GB+ RAM, 8+ CPU cores. Training the Isolation Forest/XGBoost ensemble and extracting TSFRESH features on the full SGCC dataset requires significant memory overhead.

### Environment Setup
(Note: No .env configuration is required for local execution unless overriding default Redis or FastAPI ports).
```bash
# Install Poetry dependencies
poetry install

# Optional: Export to standard requirements.txt for pip users
poetry export -f requirements.txt --output requirements.txt --without-hashes
```
### Running Tests
To verify system integrity and logic before deployment or committing changes:
```bash 
# Execute all unit and integration tests
poetry run pytest tests/ -v
```
# 📊 Dataset Acquisition
Due to licensing, the SGCC (State Grid Corporation of China) dataset must be downloaded manually.

Download the dataset from Kaggle: [SGCC Dataset (bensalem14)](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset)

Extract and place the sgcc.csv file into the designated raw data directory:
```bash
mkdir -p data/raw
mv path/to/downloaded/sgcc.csv data/raw/
```

# ⚙️ Execution Pipeline
### 1. Training & Quantization

```bash
# Execute the full pipeline: Data Ingestion → Feature Extraction → Model Training
poetry run python scripts/train_models.py --config config/pipeline_config.yaml --output models/

# Download and quantize the SLM for edge-optimized reporting
poetry run python scripts/quantize_llm.py --model phi-3-mini
```
### 2. Manual Inference Stack

```bash
redis-server &
celery -A src.workers.tasks worker --loglevel=info -Q math_queue,llm_queue &
uvicorn src.api.main:app --port 8000
```

# 🛠️ Production & MLOps Features
- **Data Drift Monitoring:** Implements Population Stability Index (PSI) and KS-tests to detect seasonal distribution shifts, triggering alerts if PSI > 0.2.

- **Edge-Optimized:** 4-bit GGUF quantization allows the Phi-3-mini reasoning engine to run efficiently on 4GB RAM edge devices.

- **Explainability (XAI):** Integrated SHAP values for every prediction, ensuring field inspection teams understand the why behind a fraud alert.

- **Automated CI/CD:** GitHub Actions workflow enforces unit/integration testing, drift evaluation, and container builds on every push.

# 🎛️ Configuration
System parameters are decoupled from the codebase in config/pipeline_config.yaml:
```bash
feature_extraction:
  tsfresh_settings: "EfficientFCParameters"
  n_jobs: 4
  
detection:
  isolation_forest:
    contamination: 0.05
    n_estimators: 100
  xgboost:
    max_depth: 6
    learning_rate: 0.05
    scale_pos_weight: 10

llm:
  model: "phi-3-mini-4k-instruct"
  quantization: "Q4_K_M"
  max_tokens: 256
```

# Kubernetes
```bsah
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml  # Auto-scaling
```
# 📚 Documentation
- Architecture Deep Dive
- API Reference
- Deployment Guide
- MLOps Setup


# 🙏 Acknowledgments
- SGCC Dataset: State Grid Corporation of China
- TSFRESH: Blue Yonder GmbH
- Hybrid Architecture: Based on "A Hybrid Machine Learning Framework for Electricity Fraud Detection" 


## Key Production-Ready Elements

1. **Efficient Architecture**: Uses TSFRESH for feature extraction (not raw text), Isolation Forest for anomaly scoring, XGBoost for classification, and SLM **only** for report generation.

2. **Quantization**: Includes GGUF/AWQ quantization scripts for edge deployment on utility company servers.

3. **Drift Monitoring**: Implements PSI (Population Stability Index) and KS-tests to detect seasonal changes and infrastructure updates.

4. **CI/CD**: GitHub Actions workflow with automated testing, training, drift detection, and containerized deployment.

5. **Explainability**: SHAP integration for every prediction, enabling transparency for field inspection teams.

6. **Class Imbalance Handling**: SMOTETomek resampling and XGBoost `scale_pos_weight` for the highly imbalanced SGCC dataset.

This architecture avoids the "textualizing time-series" anti-pattern while delivering the interpretability benefits of SLMs through structured prompting.
