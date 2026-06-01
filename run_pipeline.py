"""
run_pipeline.py
---------------
End-to-end Titanic ML pipeline (with Phase 2 Hyperparameter Tuning).

Usage (from project root with .venv activated):
    python run_pipeline.py

Phases:
    1. Load data
    2. Feature engineering (Title, FamilySize, Deck, TicketGroupSize, Sex_Pclass …)
    3. Preprocessing (impute, encode)
    4. Feature selection
    5. Baseline cross-validation (Logistic Regression, RF, GB, XGBoost)
    6. Phase 2 – Hyperparameter Tuning (RF, GB, XGBoost via RandomizedSearchCV)
    7. Evaluate tuned ensemble (confusion matrix, feature importance, ROC, comparison)
    8. Save best model → models/
    9. Generate final Kaggle submission → submissions/best_submission.csv
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Set matplotlib backend to Agg to prevent blocking in headless environments
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

from src.helpers       import load_data, save_submission, display_info, set_seed
from src.preprocessing import clean
from src.features      import engineer, select_features
from src.model         import (
    build_baseline_model,
    build_tuned_voting_classifier,
    train_model,
    predict,
    cross_validate_model,
    save_model,
)
from src.tuning import run_all_tuning
from src.stacking import build_stacking_classifier
from src.evaluate import (
    accuracy_score,
    print_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    roc_auc,
    compare_models,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
set_seed(42)

DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
TRAIN_PATH      = os.path.join(DATA_DIR, "train.csv")
TEST_PATH       = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, "best_submission.csv")

# Number of random parameter configs to try per model (increase for better results,
# but longer runtime). 40 gives a good speed/quality trade-off.
TUNING_N_ITER = 40

print("\n" + "="*60)
print("  Titanic ML -- Full Pipeline  (Phase 2: Tuning)")
print("="*60)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("\n[1/9] Loading data...")
train_raw = load_data(TRAIN_PATH)
test_raw  = load_data(TEST_PATH)
display_info(train_raw, "Train (raw)")
display_info(test_raw,  "Test  (raw)")

test_ids = test_raw["PassengerId"].copy()

# ---------------------------------------------------------------------------
# 2+3. Feature engineering then preprocessing
#      Order matters: engineer() must run BEFORE clean() drops Name/Ticket/Cabin
# ---------------------------------------------------------------------------
print("\n[2/9] Feature engineering...")
train_fe = engineer(train_raw.copy())
test_fe  = engineer(test_raw.copy())

print("\n[3/9] Preprocessing...")
train_clean = clean(train_fe)
test_clean  = clean(test_fe)

print(f"  Train: {train_clean.shape}  |  Test: {test_clean.shape}")

# ---------------------------------------------------------------------------
# 4. Feature selection
# ---------------------------------------------------------------------------
print("\n[4/9] Selecting features...")
feature_cols = select_features(train_clean)
print(f"  {len(feature_cols)} features: {feature_cols}")

TARGET  = "Survived"
X_train = train_clean[feature_cols]
y_train = train_clean[TARGET]
X_test  = test_clean[feature_cols]

# ---------------------------------------------------------------------------
# 5. Baseline cross-validation (for comparison with tuned results)
# ---------------------------------------------------------------------------
print("\n[5/9] Baseline cross-validation (5-fold stratified)...")

BASELINE_TYPES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

baseline_cv = {}
for mtype in BASELINE_TYPES:
    label = mtype.replace("_", " ").title()
    print(f"\n  -- {label} (baseline) --")
    model  = build_baseline_model(mtype)
    result = cross_validate_model(model, X_train, y_train, cv=5)
    baseline_cv[label] = result

best_baseline_label = max(baseline_cv, key=lambda k: baseline_cv[k]["mean"])
best_baseline_cv    = baseline_cv[best_baseline_label]["mean"]
print(f"\n  [BASELINE BEST] {best_baseline_label}  (CV = {best_baseline_cv:.4f})")

# ---------------------------------------------------------------------------
# 6. Phase 2 & 3 – Hyperparameter Tuning & Stacking
# ---------------------------------------------------------------------------
print(f"\n[6/9] Phase 2 & 3: Tuning and Stacking  (n_iter={TUNING_N_ITER} per model)...")
tuned_estimators = run_all_tuning(X_train, y_train, n_iter=TUNING_N_ITER, cv=5)

# Build and cross-validate the tuned soft-voting ensemble
print("\n  -- Tuned Soft-Voting Ensemble --")
tuned_ensemble = build_tuned_voting_classifier(tuned_estimators)
ensemble_result = cross_validate_model(tuned_ensemble, X_train, y_train, cv=5)

# Phase 3: Stacking Classifier
print("\n  -- Tuned Stacking Classifier --")
tuned_stacker = build_stacking_classifier(tuned_estimators)
stacker_result = cross_validate_model(tuned_stacker, X_train, y_train, cv=5)

# Compare tuned ensemble vs stacker vs best baseline
print(f"\n  Baseline best  : {best_baseline_cv:.4f}")
print(f"  Tuned ensemble : {ensemble_result['mean']:.4f}")
print(f"  Tuned stacker  : {stacker_result['mean']:.4f}")

if stacker_result['mean'] > ensemble_result['mean']:
    final_model_name = "Tuned Stacker"
    final_model_unfitted = tuned_stacker
    final_cv = stacker_result['mean']
else:
    final_model_name = "Tuned Ensemble"
    final_model_unfitted = tuned_ensemble
    final_cv = ensemble_result['mean']

print(f"\n  [WINNER] {final_model_name} (Lift vs baseline: {(final_cv - best_baseline_cv)*100:+.2f} pp)")

# Fit the winning ensemble on all training data
print(f"\n  Fitting {final_model_name} on full training set...")
best_model = train_model(final_model_unfitted, X_train, y_train)

# Build comparison dict (baseline + tuned models)
all_cv_results = dict(baseline_cv)
all_cv_results["Tuned Ensemble"] = ensemble_result
all_cv_results["Tuned Stacker"] = stacker_result

# ---------------------------------------------------------------------------
# 7. Evaluate best model on training data (diagnostic)
# ---------------------------------------------------------------------------
print("\n[7/9] Evaluating tuned ensemble on training data...")
y_train_pred = predict(best_model, X_train)
accuracy_score(y_train, y_train_pred)
print_classification_report(y_train, y_train_pred)

plot_confusion_matrix(y_train, y_train_pred, save=True)
# Feature importance not available for VotingClassifier directly;
# show it for the tuned RF sub-model instead
rf_sub = best_model.estimators_[0]  # RF is index 0
plot_feature_importance(rf_sub, feature_cols, top_n=len(feature_cols), save=True)
compare_models(all_cv_results, save=True)

if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_train)[:, 1]
    roc_auc(y_train, y_prob, save=True)

# ---------------------------------------------------------------------------
# 8. Save model
# ---------------------------------------------------------------------------
print("\n[8/9] Saving tuned ensemble...")
save_model(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

# ---------------------------------------------------------------------------
# 9. Generate final Kaggle submission
# ---------------------------------------------------------------------------
print("\n[9/9] Generating submission...")
y_test_pred = predict(best_model, X_test)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived":    y_test_pred.astype(int),
})
save_submission(submission, SUBMISSION_PATH)

print("\n" + "="*60)
print("  [DONE] Pipeline complete!")
print(f"  Winning Model     : {final_model_name} (CV = {final_cv:.4f})")
print(f"  Submission        -> {SUBMISSION_PATH}")
print("="*60 + "\n")
