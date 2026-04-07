# 🏠 House Price Prediction — Ames Housing Dataset

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

A complete end-to-end machine learning project to predict residential home sale prices using the Ames, Iowa housing dataset from Kaggle. This project covers the full ML lifecycle: data acquisition, exploratory analysis, feature engineering, model training, evaluation, and deployment.

---

## 📋 Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Project Architecture](#2-project-architecture)
3. [Setup & Installation](#3-setup--installation)
4. [Data](#4-data)
5. [Workflow](#5-workflow)
6. [Results](#6-results)
7. [References](#7-references)

---

## 1. Problem Definition

### 🎯 Objective
Predict the **final sale price** of a residential home in Ames, Iowa, given 79 explanatory variables describing physical attributes, location, quality ratings, and other characteristics of the property.

### 📌 Problem Type
- **Task:** Supervised Regression
- **Target variable:** `SalePrice` (continuous, in USD)
- **Evaluation metric:** Root Mean Squared Logarithmic Error (RMSLE)

> Using the log transformation of SalePrice is intentional — it penalizes errors on cheaper homes and expensive homes equally, which reflects real-world business fairness.

### 📊 Dataset
| Property | Detail |
|----------|--------|
| Source | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Origin | Ames, Iowa housing data compiled by Dean De Cock (2011) |
| Training samples | 1,460 |
| Test samples | 1,459 |
| Features | 79 (36 numeric, 43 categorical) |
| Target | `SalePrice` (USD) |

### ❓ Key Questions to Answer
- Which features have the strongest influence on house prices?
- How do location, size, and quality interact to drive price?
- Can we build a model that generalizes well on unseen data?
- Which ML algorithm performs best on this dataset?

### 🚧 Constraints & Assumptions
- Data is limited to Ames, Iowa (2006–2010) — model will not generalize to other cities without retraining.
- We assume the data is a representative sample of the market during that period.
- Outliers in `GrLivArea` (very large homes sold cheaply) will be investigated and potentially removed.

### ✅ Success Criteria
| Metric | Target |
|--------|--------|
| RMSLE (Kaggle leaderboard) | < 0.13 |
| R² on validation set | > 0.90 |

---

## 2. Project Architecture

```
house-price-prediction/
│
├── data/
│   ├── raw/                  # Original, immutable Kaggle data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_description.txt
│   └── processed/            # Cleaned & engineered feature sets
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── notebooks/                # Jupyter notebooks (numbered by step)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── src/                      # Reusable Python modules
│   ├── data/
│   │   └── load_data.py      # Data loading & splitting utilities
│   ├── features/
│   │   └── build_features.py # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py          # Model training scripts
│   │   └── predict.py        # Inference / submission generation
│   └── visualization/
│       └── plots.py          # Reusable plotting functions
│
├── reports/
│   └── figures/              # Saved charts and plots from EDA
│
├── tests/                    # Unit tests for src/ modules
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. Setup & Installation

### Prerequisites
- Python 3.10+
- A [Kaggle account](https://www.kaggle.com) with API token configured

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the Data

```bash
# Using Kaggle CLI (requires ~/.kaggle/kaggle.json)
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw/
unzip data/raw/house-prices-advanced-regression-techniques.zip -d data/raw/
```

Or download manually from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place files in `data/raw/`.

---

## 4. Data

| File | Description |
|------|-------------|
| `data/raw/train.csv` | Training set with SalePrice labels |
| `data/raw/test.csv` | Test set for Kaggle submission |
| `data/raw/data_description.txt` | Full description of all 79 features |

Key feature groups:
- **Size:** `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `LotArea`
- **Quality:** `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`
- **Location:** `Neighborhood`, `MSZoning`
- **Age:** `YearBuilt`, `YearRemodAdd`
- **Extras:** `Fireplaces`, `PoolArea`, `Fence`

---

## 5. Workflow

| Step | Description | Notebook |
|------|-------------|----------|
| ✅ Step 1 | Data Acquisition | — |
| ✅ Step 2 | Repository Setup | — |
| ✅ Step 3 | Problem Definition | README |
| 🔄 Step 4 | Exploratory Data Analysis | `01_eda.ipynb` |
| ⬜ Step 5 | Preprocessing & Feature Engineering | `02_preprocessing.ipynb` |
| ⬜ Step 6 | Model Training | `03_modeling.ipynb` |
| ⬜ Step 7 | Evaluation & Interpretation | `04_evaluation.ipynb` |
| ⬜ Step 8 | Finalize & Publish | — |

---

## 6. Results

> 🚧 This section will be updated as the project progresses.

| Model | Val RMSLE | Val R² |
|-------|-----------|--------|
| Baseline (Mean) | — | — |
| Ridge Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |
| Ensemble | — | — |

---

## 7. References

- De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data Set.* Journal of Statistics Education.
- [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Data Description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
