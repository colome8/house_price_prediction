"""Preprocessing pipeline for the Ames Housing dataset.

Every step lives inside a single sklearn ``Pipeline`` so that:
  * statistics (medians, skewness, scaling, categories) are learned only from
    the training data -> no data leakage during cross-validation;
  * the exact same transformations are applied to the test set / new houses.

The pipeline receives the raw DataFrame (as read from ``train.csv``) and
returns a fully numeric matrix ready for modeling.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Domain knowledge (from data_description.txt)
# ---------------------------------------------------------------------------

# NaN means "the house does not have this feature", not "unknown".
NONE_CAT_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
ZERO_NUM_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

# Ordinal categories -> integers that preserve their natural order.
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    **{col: QUALITY_MAP for col in [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
    ]},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
}

# Dropped: identifier, and a feature that is constant in practice
# (all houses but one in train have AllPub; all of them in test).
DROP_COLS = ["Id", "Utilities"]


# ---------------------------------------------------------------------------
# Stateless steps (no parameters learned from data)
# ---------------------------------------------------------------------------

def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaNs that mean 'feature absent' with 'None' / 0."""
    df = df.copy()
    for col in NONE_CAT_COLS:
        df[col] = df[col].fillna("None")
    for col in ZERO_NUM_COLS:
        df[col] = df[col].fillna(0)
    # No garage -> use the house's build year instead of an impossible year 0.
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
    # data_description.txt: "Assume typical unless deductions are warranted".
    df["Functional"] = df["Functional"].fillna("Typ")
    return df


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordered categories (Po < Fa < TA < Gd < Ex, ...) to integers."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping).astype(float)  # unseen/NaN stay NaN -> imputed later
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features."""
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
                          + df["ScreenPorch"] + df["WoodDeckSF"])
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["YearsSinceRemod"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["QualxSF"] = df["OverallQual"] * df["GrLivArea"]

    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    # Numeric codes that are really categories.
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)

    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


# ---------------------------------------------------------------------------
# Stateful steps (learn parameters in fit, only from training data)
# ---------------------------------------------------------------------------

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute ``target`` with its median within each ``group`` (e.g. LotFrontage
    by Neighborhood), falling back to the global median for unseen groups."""

    def __init__(self, target="LotFrontage", group="Neighborhood"):
        self.target = target
        self.group = group

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group)[self.target].median()
        self.global_median_ = X[self.target].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.group].map(self.group_medians_).fillna(self.global_median_)
        X[self.target] = X[self.target].fillna(fill)
        return X


class SkewCorrector(BaseEstimator, TransformerMixin):
    """Apply log1p to non-negative numeric columns whose skewness (measured on
    the training data) exceeds ``threshold``."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        skew = X.skew()
        self.skewed_cols_ = [c for c in X.columns
                             if skew[c] > self.threshold and X[c].min() >= 0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.skewed_cols_] = np.log1p(X[self.skewed_cols_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(skew_threshold: float = 0.75) -> Pipeline:
    """Raw DataFrame in -> numeric DataFrame out."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("unskew", SkewCorrector(threshold=skew_threshold)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        # pandas >= 3 uses a dedicated 'str' dtype, so select "everything non-numeric".
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False).set_output(transform="pandas")

    # Earlier steps already work on DataFrames, so only the final step needs set_output.
    return Pipeline([
        ("structural_na", FunctionTransformer(fill_structural_missing)),
        ("lot_frontage", GroupMedianImputer("LotFrontage", "Neighborhood")),
        ("ordinals", FunctionTransformer(encode_ordinals)),
        ("features", FunctionTransformer(add_features)),
        ("columns", columns),
    ])


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 2 documented outliers (huge living area, very low price).
    Applied to training data only, never to the test set."""
    mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 200_000)
    return df.loc[~mask].reset_index(drop=True)
