
# E-Commerce Purchase Intention Prediction

Predicting whether an online shopping session will result in a purchase (`Revenue`) using behavioral, session, and page-metric data. Built as the Midterm Project for the **DataTalks.Club Machine Learning Zoomcamp (2026 Cohort)**.

## Problem Statement

E-commerce platforms generate high volumes of browsing traffic, yet only a small fraction of sessions convert into completed transactions (~15.5%). By predicting purchase intention from session-level metrics (such as product page views, time spent, bounce/exit rates, visitor type, and seasonality), marketing and product teams can trigger targeted interventions—such as personalized promotions, dynamic checkout nudges, or cart-abandonment recovery—in real time.

## Dataset

* **Source:** [Online Shoppers Purchasing Intention Dataset (UCI Machine Learning Repository, ID: 468)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
* **Size:** 12,330 sessions, 17 input features, and 1 binary target (`Revenue`: `True` / `False`).
* **Local Path:** `data/raw/online_shoppers_intention.csv`

## Project Structure

```text
ecommerce-purchase-intention-prediction/
├── data/
│   ├── raw/                  # Raw dataset (online_shoppers_intention.csv)
│   └── processed/            # Cleaned/split datasets (if saved)
├── models/                   # Serialized model artifacts
├── notebooks/
│   └── notebook.ipynb        # Exploratory Data Analysis (EDA), feature analysis & model tuning
├── train.py                  # Final model training script
├── predict.py                # FastAPI inference web service
├── test.py                   # API smoke-test script
├── Dockerfile                # Container configuration
├── pyproject.toml            # Project dependencies (managed via uv)
├── uv.lock                   # Reproducible dependency lockfile
└── README.md                 # Project documentation
```
