# Electricity Theft Detection via Hybrid SLM + Time-Series Anomaly Detection

[![CI](https://github.com/amiraghabiglou/electricity-theft-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/amiraghabiglou/electricity-theft-detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready ML system designed to detect anomalous electricity consumption patterns indicative of energy theft. This project bridges the gap between robust statistical time-series analysis and modern Small Language Models (SLMs) to provide highly accurate, interpretable fraud investigation reports.
## 🏗️ Architecture

This system addresses the **critical weaknesses** of naive "text-as-input" approaches by using a **hybrid architecture**:

| Component | Technology | Role |
|-----------|-----------|------|
| **Feature Extraction** | TSFRESH | Extracts statistical features (trend, variance, entropy) from raw time-series matrices. |
| **Anomaly Detection** | XGBoost | High-precision fraud detection relying on mathematically shaped data. |
| **Message Broker** | Redis & Celery | Decouples heavy mathematical/LLM workloads from the API to maintain responsiveness. |
| **Reasoning Engine** | Phi-3 (4-bit GGUF) | Generates natural language investigation reports based strictly on XGBoost/SHAP outputs. |

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
The fastest way to evaluate this system is via Docker. The stack runs out-of-the-box using the defaults in `docker-compose.yml`.

### Prerequisites
- Docker and Docker Compose.

- 8GB+ RAM.

- Python 3.10+ and Poetry (for local data prep)

### Installation
1. Clone the repository
```bash 
git clone git@github.com:amiraghabiglou/electricity-theft-detection.git
cd electricity-theft-detection
```
2. Install dependencies (Required for local execution)
```bash 
poetry install
```
3. Create necessary directories
```bash 
mkdir -p models data/raw data/processed
```
4. 📊 Dataset Acquisition
Due to licensing, the SGCC (State Grid Corporation of China) dataset must be downloaded manually.

Download the dataset from Kaggle: [SGCC Dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset)

Extract, rename (manually data set.csv to sgcc.csv) and place the sgcc.csv file into the designated raw data directory:
```bash
mv path/to/downloaded/sgcc.csv data/raw/
```
5. Process the data (Ensure sgcc.csv is in data/raw/ as per the Dataset guide)

**Note:** To train the full production model, omit the --sample flag
```bash
poetry run python -m src.pipeline.data_pipeline --input data/raw/sgcc.csv --output data/processed/features.parquet --sample 1000
```
6. Train the mathematical anomaly detection model and generate hybrid_detector.joblib:
```bash
poetry run python scripts/train_models.py \
    --features data/processed/features.parquet \
    --model-output models/hybrid_detector.joblib \
    --metrics-output models/metrics.json
```
7. Download the Reasoning Engine (SLM)
You must download the actual binary weights for the Generative AI model. We use a 4-bit quantized version of Microsoft's Phi-3 (~2.4GB).
```bash 
curl -L -o models/phi-3-q4.gguf [https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf)
```
(Verify the download is ~2.4GB using ls -lh models/. If it is 1KB, the download failed).

8. Boot the Infrastructure
```bash
docker-compose up -d --build
```
10. Verify the API is running
```bash
curl http://localhost:8000/health
```

# ⚡ How to Use the API
Because the ML pipeline is computationally heavy, the API is entirely asynchronous.

**Step 1:** Submit a Fraud Detection Job
Send a POST request with the raw daily consumption data. Notice the sudden drop to 0.0 in the data below, simulating a meter bypass.

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{
           "consumers": [
             {
               "consumer_id": "THEFT_999",
               "consumption_data": [6.1, 6.3, 5.9, 6.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
             }
           ]
         }'
```
**Response:** You will receive a 202 Accepted status with a job_id:
```bash
{"job_id":"<YOUR_JOB_ID>","status":"Processing"}
```

**Step 2:** Retrieve the Results & LLM Report
Wait 10-30 seconds for the math and LLM workers to process the queue, then poll the results endpoint using your specific job_id:

```bash
curl http://localhost:8000/results/<YOUR_JOB_ID>
```
**Response:** If fraud is detected, the JSON will include the mathematical SHAP explanations alongside a generated natural language report from the Phi-3 model.

💡 UI Tip: You can also test these endpoints interactively directly in your browser by navigating to the Swagger UI: http://localhost:8000/docs
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
