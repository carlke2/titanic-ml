"""
tune.py
-------
Hyperparameter tuning utilities using scikit-learn's GridSearchCV
and RandomizedSearchCV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold


def grid_search(
    model,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Run an exhaustive grid search over the provided parameter grid.

    Parameters
    ----------
    model      : sklearn estimator (unfitted)
    param_grid : dict  Parameter names → values to try.
    X          : pd.DataFrame  Feature matrix.
    y          : pd.Series     Target labels.
    cv         : int  Cross-validation folds (default 5).
    scoring    : str  Metric to optimise (default 'accuracy').
    n_jobs     : int  Parallel jobs (-1 = all CPUs).
    verbose    : int  Verbosity level.

    Returns
    -------
    Fitted GridSearchCV object.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    gs.fit(X, y)
    print(f"\n[OK] Grid search complete.")
    print(f"   Best params : {gs.best_params_}")
    print(f"   Best score  : {gs.best_score_:.4f}")
    return gs


def random_search(
    model,
    param_distributions: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """Run a randomised search over parameter distributions.

    More efficient than grid search when the parameter space is large.

    Parameters
    ----------
    model                : sklearn estimator (unfitted)
    param_distributions  : dict  Parameter names → distributions or lists.
    X                    : pd.DataFrame  Feature matrix.
    y                    : pd.Series     Target labels.
    n_iter               : int  Number of random combinations to try (default 30).
    cv                   : int  Cross-validation folds (default 5).
    scoring              : str  Metric to optimise (default 'accuracy').
    n_jobs               : int  Parallel jobs.
    verbose              : int  Verbosity level.

    Returns
    -------
    Fitted RandomizedSearchCV object.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        random_state=42,
    )
    rs.fit(X, y)
    print(f"\n[OK] Random search complete ({n_iter} iterations).")
    print(f"   Best params : {rs.best_params_}")
    print(f"   Best score  : {rs.best_score_:.4f}")
    return rs


def get_best_params(search_result) -> dict:
    """Extract best parameters and score from a fitted search object.

    Parameters
    ----------
    search_result : GridSearchCV or RandomizedSearchCV (fitted)

    Returns
    -------
    dict with keys 'best_params' and 'best_score'.
    """
    return {
        "best_params": search_result.best_params_,
        "best_score": search_result.best_score_,
        "best_estimator": search_result.best_estimator_,
    }
