"""Build a complete house from a few user inputs and predict its price."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

SALE_YEAR = 2010  # last year in the data: the model knows the 2006-2010 market
SQFT_PER_GARAGE_CAR = 240  # approx. median garage area per car in the data

# Basement / garage descriptive features: NaN means "doesn't have one".
BSMT_CAT = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
GARAGE_CAT = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]


def load_artifacts():
    model = joblib.load(MODEL_DIR / "final_model.joblib")
    with open(MODEL_DIR / "typical_house.json") as f:
        typical = json.load(f)
    return model, typical


def build_house(typical: dict, *, neighborhood: str, overall_qual: int, overall_cond: int,
                living_area: float, floors: int, basement_area: float, lot_area: float,
                year_built: int, full_baths: int, garage_cars: int, kitchen_qual: str,
                fireplaces: int) -> pd.DataFrame:
    """Start from a typical house and overwrite the features given by the user,
    keeping related features consistent (e.g. no garage -> no garage quality)."""
    h = dict(typical)
    h.update({
        "Id": 0,
        "Neighborhood": neighborhood,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "LotArea": lot_area,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "YrSold": SALE_YEAR,
        "FullBath": full_baths,
        "KitchenQual": kitchen_qual,
        "Fireplaces": fireplaces,
        # Living area split across floors
        "GrLivArea": living_area,
        "1stFlrSF": living_area if floors == 1 else living_area / 2,
        "2ndFlrSF": 0 if floors == 1 else living_area / 2,
        "LowQualFinSF": 0,
        "HouseStyle": "1Story" if floors == 1 else "2Story",
        # Dwelling type code: 20 = 1-story, 60 = 2-story (1946 & newer); 30 / 70 = older
        "MSSubClass": (20 if floors == 1 else 60) if year_built >= 1946 else (30 if floors == 1 else 70),
        # Basement: half finished, half unfinished
        "TotalBsmtSF": basement_area,
        "BsmtFinSF1": basement_area / 2,
        "BsmtUnfSF": basement_area / 2,
        "BsmtFinSF2": 0,
        # Garage
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * SQFT_PER_GARAGE_CAR,
        "GarageYrBlt": year_built,
    })
    if basement_area == 0:
        h.update({c: None for c in BSMT_CAT} | {"BsmtFullBath": 0, "BsmtHalfBath": 0})
    if garage_cars == 0:
        h.update({c: None for c in GARAGE_CAT})
    if fireplaces == 0:
        h["FireplaceQu"] = None
    elif h.get("FireplaceQu") is None:
        h["FireplaceQu"] = "TA"

    df = pd.DataFrame([h])
    # Missing categoricals must be real NaN (not the string "None") for the pipeline.
    return df.replace({None: np.nan})


def predict_price(model, house: pd.DataFrame) -> float:
    """Predicted sale price in USD."""
    return float(np.expm1(model.predict(house)[0]))
