"""SHAP explanations for the final Voting(Lasso + XGBoost) model.

The final model predicts the plain average of two models:
    f(x) = 0.5 * lasso(x) + 0.5 * xgb(x)
SHAP values are additive, so the ensemble's SHAP values are exactly the
average of each model's SHAP values:
  * Lasso (linear):  phi_j = coef_j * (x_j - mean(x_j))
  * XGBoost (trees): exact TreeSHAP via ``shap.TreeExplainer``
"""

import numpy as np
import pandas as pd
import shap


def ensemble_shap(final_model, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-price scale) for each one-hot / numeric model feature."""
    lasso = final_model.named_estimators_["lasso"]
    xgb = final_model.named_estimators_["xgb"]

    # Both pipelines learned identical preprocessing from the same data.
    Xt = lasso.named_steps["prep"].transform(X)

    coef = lasso.named_steps["model"].coef_
    mean = Xt.values.mean(axis=0)
    shap_lasso = (Xt.values - mean) * coef
    base_lasso = lasso.named_steps["model"].intercept_ + mean @ coef

    tree = shap.TreeExplainer(xgb.named_steps["model"])
    shap_xgb = tree.shap_values(Xt)
    base_xgb = tree.expected_value

    return shap.Explanation(
        values=0.5 * shap_lasso + 0.5 * shap_xgb,
        base_values=np.full(len(Xt), 0.5 * base_lasso + 0.5 * base_xgb),
        data=Xt.values,
        feature_names=list(Xt.columns),
    )


def group_by_original_feature(expl: shap.Explanation, final_model, X: pd.DataFrame) -> shap.Explanation:
    """Sum the SHAP values of one-hot columns back into their original feature
    (e.g. all ``Neighborhood_*`` -> ``Neighborhood``), so each feature gets a
    single, readable contribution. ``data`` holds human-readable values
    (before scaling / one-hot) for plots."""
    prep = final_model.named_estimators_["lasso"].named_steps["prep"]
    readable = prep[:-1].transform(X)  # after cleaning + feature engineering, before scaling
    cat_cols = next(cols for name, _, cols in prep.named_steps["columns"].transformers_ if name == "cat")

    def original(col):
        for c in cat_cols:
            if col.startswith(c + "_"):
                return c
        return col

    groups = pd.Series([original(c) for c in expl.feature_names])
    names = list(dict.fromkeys(groups))
    values = np.column_stack([expl.values[:, (groups == g).values].sum(axis=1) for g in names])

    # Categorical values cannot be used to color plots -> NaN (shown in gray).
    data = np.column_stack([
        pd.to_numeric(readable[g], errors="coerce").to_numpy(dtype=float) if g not in cat_cols
        else np.full(len(readable), np.nan)
        for g in names
    ])
    return shap.Explanation(
        values=values, base_values=expl.base_values, data=data, feature_names=names,
        display_data=readable[names].to_numpy(),  # raw values, incl. category labels
    )
