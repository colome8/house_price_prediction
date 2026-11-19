"""Model definitions. Every model is wrapped in a Pipeline together with the
preprocessor, so ``fit``/``predict`` work directly on the raw DataFrame.

Hyperparameters were selected in ``notebooks/03_modeling.ipynb`` with 5-fold
cross-validation on log(SalePrice).
"""

from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def with_preprocessing(model) -> Pipeline:
    """Raw DataFrame -> preprocessing -> model."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def make_ridge(alpha: float = 20) -> Pipeline:
    return with_preprocessing(Ridge(alpha=alpha))


def make_lasso(alpha: float = 5e-4) -> Pipeline:
    return with_preprocessing(Lasso(alpha=alpha, max_iter=50_000))


def make_lgbm() -> Pipeline:
    # Many shallow trees + small learning rate + row/column subsampling:
    # a conservative setup that generalizes well on a small dataset (~1,500 rows).
    return with_preprocessing(LGBMRegressor(
        n_estimators=2000, learning_rate=0.01, num_leaves=8, min_child_samples=10,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1,
        random_state=RANDOM_STATE, verbose=-1,
    ))


def make_xgb() -> Pipeline:
    return with_preprocessing(XGBRegressor(
        n_estimators=2000, learning_rate=0.01, max_depth=3, min_child_weight=2,
        subsample=0.7, colsample_bytree=0.3, reg_lambda=1,
        random_state=RANDOM_STATE, n_jobs=-1,
    ))


def make_final_model() -> VotingRegressor:
    """Final model: average of Lasso (linear) and XGBoost (non-linear).
    Both make different kinds of errors, so averaging them reduces the error."""
    return VotingRegressor([("lasso", make_lasso()), ("xgb", make_xgb())])
