# E-Commerce Purchase Intention Prediction

Predicting whether an online shopping session will result in a purchase (`Revenue`) using behavioral, session, and page-metric data.

## Problem statement

E-commerce sites receive large amounts of browsing traffic, but only a small share of visitors actually complete a purchase. This project builds a binary classification model that estimates the probability a session will result in a purchase, based on session-level behavior such as page visits, time spent, rates of bounce/exit, campaign timing, and visitor type.

The dataset is the `Online Shoppers Purchasing Intention` dataset from the UCI ML repository. The target is `Revenue`, where `True` indicates a purchase occurred and `False` indicates no purchase.

## Key data quality and leakage note

`PageValues` is a strongly predictive, post-conversion signal because it is aggregate value from completed purchases. It reflects purchase outcome information and therefore is not available at the intended prediction time for a new session. For this reason, the final production model excludes `PageValues` from the feature set to avoid target leakage.

## Dataset

- Source: <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
- Local file: `data/raw/online_shoppers_intention.csv`
- Rows: 12,330
- Features: 17 input features
- Target: `Revenue`

## Repository structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── notebook.ipynb
├── train.py
├── predict.py
├── test.py
├── Dockerfile
├── .dockerignore
├── README.md
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Reproducibility and dependency management

This project uses `uv` for dependency management and environment reproduction. The repository is pinned to Python 3.12 in `.python-version` and `pyproject.toml`.

### Prerequisites

- Install `uv` if it is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Confirm the project root and dataset are present:

```bash
ls
ls data/raw
```

The expected dataset file is:

```text
data/raw/online_shoppers_intention.csv
```

### 1) Create the project environment

From the project root:

```bash
uv sync --python 3.12
```

This creates the isolated environment and installs the exact dependencies from `pyproject.toml` and `uv.lock`.

### 2) Train the model

Run the production training pipeline:

```bash
uv run python train.py
```

This script:

- loads the raw dataset,
- applies the finalized leakage-safe preprocessing,
- splits the data reproducibly,
- trains the selected model,
- evaluates it on the held-out test set,
- saves the trained pipeline as `models/purchase_intention_model.joblib`.

### 3) Start the prediction API

Start the FastAPI service from the project root:

```bash
uv run python -m uvicorn predict:app --host 0.0.0.0 --port 8000
```

Optional: run it in the background in a separate terminal session:

```bash
uv run python -m uvicorn predict:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

### 4) Run the smoke test

With the API running, execute:

```bash
uv run python test.py
```

This script sends a valid request to `http://localhost:8000/predict`, checks the HTTP status code, verifies the expected JSON fields, and confirms that the probability is within `[0, 1]`.

## API usage

The API exposes:

- `GET /health`
- `POST /predict`

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 2,
    "ProductRelated_Duration": 64.0,
    "BounceRates": 0.0,
    "ExitRates": 0.1,
    "SpecialDay": 0.0,
    "Month": "Feb",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

Example response:

```json
{
  "prediction": false,
  "predicted_class": false,
  "probability": 0.0017494770321775227
}
```

## Model results

The final model was evaluated on a held-out test set using behavior-based features only, excluding `PageValues` to avoid target leakage.

- Accuracy: 0.8451
- Precision: 0.5000
- Recall: 0.0864
- F1-score: 0.1473
- ROC-AUC: 0.7731

This reflects the class imbalance in the dataset and is suitable for business-facing probability estimates, not causal claims.

## Docker

```bash
docker build -t ecommerce-purchase-intention .
docker run --rm -p 8000:8000 ecommerce-purchase-intention
```

The GitHub Actions workflow in `.github/workflows/docker-build.yml` builds the image and performs a container smoke test on a GitHub-hosted runner.

## Live Cloud Deployment

The API is live on Render:

- Base URL: [https://ecommerce-purchase-intention-prediction.onrender.com](https://ecommerce-purchase-intention-prediction.onrender.com)
- Interactive Swagger UI: [https://ecommerce-purchase-intention-prediction.onrender.com/docs](https://ecommerce-purchase-intention-prediction.onrender.com/docs)

### Health check

```bash
curl -i https://ecommerce-purchase-intention-prediction.onrender.com/health
```

Verified response:

```text
HTTP/1.1 200 OK
{"status":"ok","model":"purchase_intention_model"}
```

### Make a prediction

```bash
curl -i -X POST https://ecommerce-purchase-intention-prediction.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Administrative": 2,
    "Administrative_Duration": 45.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 15,
    "ProductRelated_Duration": 420.5,
    "BounceRates": 0.01,
    "ExitRates": 0.02,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

Verified response:

```text
HTTP/1.1 200 OK
{"prediction":false,"predicted_class":false,"probability":0.19559552367618738}
```

### Render deployment settings (native Python runtime)

This project is prepared for a native Python deployment on Render without depending on Docker being installed locally.

Recommended Render settings:

- Runtime: Python
- Python version: 3.12
- Build command:

```bash
python -m pip install --upgrade pip
pip install uv
uv sync --python 3.12 --frozen
```

- Start command:

```bash
uv run python -m uvicorn predict:app --host 0.0.0.0 --port $PORT
```

- Port: use Render's `$PORT` environment variable; do not hardcode 8000 in production.

The repository also includes:

- `render.yaml` for a Render web service definition
- `runtime.txt` with `python-3.12.13`

## Notes

This project is a production-style demonstration of a binary purchase-intention model. It is intended for business analytics and conversion optimization support, not for proving causality in customer behavior. The required peer-review step should be completed before final submission.
