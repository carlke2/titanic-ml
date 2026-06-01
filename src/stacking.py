"""
stacking.py
-----------
Phase 3 – Stacking Classifiers.

Builds a StackingClassifier using the tuned base models (RF, GB, XGBoost)
as level-0 estimators and a Logistic Regression as the level-1 meta-learner.

The meta-learner is trained on out-of-fold predictions from the base models,
so it learns *how* each model is wrong and blends them intelligently.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_stacking_classifier(tuned_estimators: dict) -> StackingClassifier:
    """Build a StackingClassifier using pre-tuned base models.

    Level-0 estimators:  tuned RF, GB, and XGBoost (cloned so they are unfitted).
    Level-1 meta-learner: Logistic Regression (with standard scaling inside a pipeline).

    Parameters
    ----------
    tuned_estimators : dict
        Dict with keys 'rf', 'gb', 'xgb' mapping to fitted estimators
        (as returned by src.tuning.run_all_tuning).

    Returns
    -------
    Unfitted StackingClassifier ready to be passed to cross_validate_model / train_model.
    """
    rf  = clone(tuned_estimators["rf"])
    gb  = clone(tuned_estimators["gb"])
    xgb = clone(tuned_estimators["xgb"])

    base_estimators = [
        ("rf",  rf),
        ("gb",  gb),
        ("xgb", xgb),
    ]

    # Logistic Regression as meta-learner; wrap in a pipeline so features are scaled.
    meta_learner = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        # Use 5-fold cross-validation to generate out-of-fold meta-features.
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        # Pass predicted probabilities (not just class labels) to the meta-learner.
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,   # only meta-features go to the LR, not the raw features
    )
    return stacker


def cross_validate_stacker(
    stacker: StackingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict:
    """Cross-validate the stacking classifier and print a summary.

    Parameters
    ----------
    stacker  : StackingClassifier (unfitted).
    X        : Feature matrix.
    y        : Target labels.
    cv       : Number of outer folds.
    scoring  : Scoring metric.

    Returns
    -------
    dict with keys 'scores', 'mean', 'std'.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(stacker, X, y, cv=skf, scoring=scoring, n_jobs=1)
    result = {"scores": scores, "mean": scores.mean(), "std": scores.std()}
    print(f"  CV {cv}-fold {scoring}: {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"  Individual folds: {[f'{s:.4f}' for s in scores]}")
    return result
