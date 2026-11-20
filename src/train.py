"""Train the final model on the full training set and save the artifacts
used by the Streamlit app.

Usage (from the project root):
    python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import make_final_model
from src.preprocessing import NONE_CAT_COLS, remove_outliers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
CATEGORICAL_CODES = ["MSSubClass", "MoSold"]


def typical_house(df: pd.DataFrame) -> dict:
    """A 'typical' house: median for numeric features, most frequent value for
    categorical ones. Used to fill the features the app does not ask for.
    For structural features (pool, fence, ...) the most frequent value is
    "doesn't have it" (NaN), which is kept."""
    house = {}
    for col in df.columns:
        if col in CATEGORICAL_CODES:  # numeric codes that are really categories
            house[col] = int(df[col].mode().iloc[0])
        elif pd.api.types.is_numeric_dtype(df[col]):
            house[col] = float(df[col].median())
        elif col in NONE_CAT_COLS:
            mode = df[col].value_counts(dropna=False).idxmax()
            house[col] = None if pd.isna(mode) else str(mode)
        else:
            house[col] = str(df[col].mode().iloc[0])
    return house


def main():
    train = remove_outliers(pd.read_csv(DATA))
    y = np.log1p(train.pop("SalePrice"))

    print(f"Training final model on {len(train)} houses...")
    model = make_final_model().fit(train, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "final_model.joblib")
    with open(MODEL_DIR / "typical_house.json", "w") as f:
        json.dump(typical_house(train.drop(columns=["Id"])), f, indent=2)

    print(f"Saved model and typical house to {MODEL_DIR}")


if __name__ == "__main__":
    main()
