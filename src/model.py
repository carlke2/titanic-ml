"""
model.py
--------
Model training, prediction, cross-validation, and persistence functions.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_baseline_model(model_type: str = "random_forest") -> object:
    """Create and return an untrained scikit-learn classifier.

    Parameters
    ----------
    model_type : str
        One of 'random_forest', 'logistic_regression', 'gradient_boosting', 'xgboost', 'voting'.
        Default is 'random_forest'.

    Returns
    -------
    Untrained sklearn estimator.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "voting":
        rf  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                            eval_metric="logloss", random_state=42, n_jobs=-1)
        return VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("xgb", xgb)],
            voting="soft",
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         f"Choose from: random_forest, logistic_regression, gradient_boosting, xgboost, voting.")


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the model on training data.

    Parameters
    ----------
    model : sklearn estimator
    X_train : pd.DataFrame  Feature matrix.
    y_train : pd.Series     Target labels.

    Returns
    -------
    Fitted model.
    """
    model.fit(X_train, y_train)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Return class predictions for a given feature matrix.

    Parameters
    ----------
    model : sklearn estimator (fitted)
    X : pd.DataFrame  Feature matrix.

    Returns
    -------
    np.ndarray of predicted labels (0 or 1).
    """
    return model.predict(X)


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict:
    """Run stratified k-fold cross-validation and return a summary.

    Parameters
    ----------
    model    : sklearn estimator (unfitted clone will be used internally).
    X        : pd.DataFrame  Feature matrix.
    y        : pd.Series     Target labels.
    cv       : int  Number of folds (default 5).
    scoring  : str  Scoring metric (default 'accuracy').

    Returns
    -------
    dict with keys 'scores', 'mean', 'std'.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    result = {"scores": scores, "mean": scores.mean(), "std": scores.std()}
    print(f"  CV {cv}-fold {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Individual folds: {[f'{s:.4f}' for s in scores]}")
    return result


def save_model(model, path: str) -> None:
    """Persist a fitted model to disk using joblib.

    Parameters
    ----------
    model : fitted sklearn estimator
    path  : str  File path (e.g., 'models/rf_v1.pkl')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[saved] Model saved -> {path}")


def load_model(path: str):
    """Load a previously saved model from disk.

    Parameters
    ----------
    path : str  File path to the saved model.

    Returns
    -------
    Loaded sklearn estimator.
    """
    model = joblib.load(path)
    print(f"[loaded] Model loaded <- {path}")
    return model
