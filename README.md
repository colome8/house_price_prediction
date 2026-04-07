# House Price Prediction

An end-to-end machine learning project to predict residential home sale prices using the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

---

## Problem Definition

Given 79 features describing a residential property in Ames, Iowa (size, quality, location, age, etc.), predict its **final sale price**.

This is a **supervised regression** problem. The target variable is `SalePrice` (continuous, in USD).

---

## Project Structure

```
house-price-prediction/
├── data/               # Raw data from Kaggle (not tracked by git)
├── notebooks/          # Jupyter notebooks for each stage
├── src/                # Reusable Python scripts
├── requirements.txt
└── README.md
```

---

## Data

Downloaded from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

| File | Description |
|------|-------------|
| `train.csv` | 1,460 homes with sale prices |
| `test.csv` | 1,459 homes without sale prices |
| `data_description.txt` | Description of all 79 features |

---

## Progress

- [x] Step 1 — Get the data
- [x] Step 2 — Set up repository
- [x] Step 3 — Define the problem
- [ ] Step 4 — EDA
- [ ] Step 5 — Data processing & feature engineering
- [ ] Step 6 — Build and train model
- [ ] Step 7 — Evaluate and interpret results
- [ ] Step 8 — Finalize and publish
