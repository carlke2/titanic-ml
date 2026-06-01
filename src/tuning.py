"""
tuning.py
---------
Phase 2 – Hyperparameter Tuning.

Uses RandomizedSearchCV (no extra installs needed) with stratified 5-fold CV
to find optimal hyperparameters for Random Forest, Gradient Boosting, and XGBoost.
Returns fully fitted, tuned estimators ready for use in an ensemble.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

_RF_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [3, 4, 5, 6, 7, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf":  [1, 2, 4, 6],
    "max_features":      ["sqrt", "log2", 0.5, 0.7],
    "bootstrap":         [True, False],
}

_GB_PARAM_DIST = {
    "n_estimators":    [100, 200, 300, 500],
    "learning_rate":   [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth":       [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "subsample":       [0.7, 0.8, 0.9, 1.0],
    "max_features":    ["sqrt", "log2", None],
}

_XGB_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 500],
    "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
    "max_depth":         [2, 3, 4, 5, 6],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 1.0],
    "gamma":             [0, 0.1, 0.2, 0.3],
    "reg_alpha":         [0, 0.01, 0.1, 0.5],
    "reg_lambda":        [0.5, 1.0, 1.5, 2.0],
    "min_child_weight":  [1, 3, 5],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Find the best Random Forest hyperparameters via RandomizedSearchCV.

    Parameters
    ----------
    X            : Feature matrix.
    y            : Target labels.
    n_iter       : Number of random parameter combinations to try.
    cv           : Number of stratified CV folds.
    random_state : Seed for reproducibility.

    Returns
    -------
    Fitted RandomForestClassifier with the best found parameters.
    """
    print(f"  [RF] Searching {n_iter} random configs over {cv}-fold CV...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=_RF_PARAM_DIST,
        n_iter=n_iter,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X, y)
    print(f"  [RF] Best CV accuracy : {search.best_score_:.4f}")
    print(f"  [RF] Best params      : {search.best_params_}")
    return search.best_estimator_


def tune_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """Find the best Gradient Boosting hyperparameters via RandomizedSearchCV.

    Parameters
    ----------
    X            : Feature matrix.
    y            : Target labels.
    n_iter       : Number of random parameter combinations to try.
    cv           : Number of stratified CV folds.
    random_state : Seed for reproducibility.

    Returns
    -------
    Fitted GradientBoostingClassifier with the best found parameters.
    """
    print(f"  [GB] Searching {n_iter} random configs over {cv}-fold CV...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = GradientBoostingClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=_GB_PARAM_DIST,
        n_iter=n_iter,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X, y)
    print(f"  [GB] Best CV accuracy : {search.best_score_:.4f}")
    print(f"  [GB] Best params      : {search.best_params_}")
    return search.best_estimator_


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> XGBClassifier:
    """Find the best XGBoost hyperparameters via RandomizedSearchCV.

    Parameters
    ----------
    X            : Feature matrix.
    y            : Target labels.
    n_iter       : Number of random parameter combinations to try.
    cv           : Number of stratified CV folds.
    random_state : Seed for reproducibility.

    Returns
    -------
    Fitted XGBClassifier with the best found parameters.
    """
    print(f"  [XGB] Searching {n_iter} random configs over {cv}-fold CV...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    base = XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=_XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X, y)
    print(f"  [XGB] Best CV accuracy : {search.best_score_:.4f}")
    print(f"  [XGB] Best params      : {search.best_params_}")
    return search.best_estimator_


def run_all_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Tune all three tree-based models and return a dict of fitted estimators.

    Parameters
    ----------
    X            : Feature matrix.
    y            : Target labels.
    n_iter       : Number of random parameter combinations per model.
    cv           : Number of stratified CV folds.
    random_state : Seed for reproducibility.

    Returns
    -------
    dict with keys 'rf', 'gb', 'xgb' mapping to fitted estimators.
    """
    print("\n--- Random Forest ---")
    rf = tune_random_forest(X, y, n_iter=n_iter, cv=cv, random_state=random_state)

    print("\n--- Gradient Boosting ---")
    gb = tune_gradient_boosting(X, y, n_iter=n_iter, cv=cv, random_state=random_state)

    print("\n--- XGBoost ---")
    xgb = tune_xgboost(X, y, n_iter=n_iter, cv=cv, random_state=random_state)

    return {"rf": rf, "gb": gb, "xgb": xgb}
