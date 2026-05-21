"""
run_pipeline.py
---------------
End-to-end Titanic ML pipeline.

Usage (from project root with .venv activated):
    python run_pipeline.py

Phases:
    1. Load data
    2. Preprocess (impute missing values, encode categoricals)
    3. Feature engineering (Title, FamilySize, IsAlone, AgeBand, FareBand)
    4. Train & cross-validate: Logistic Regression, Random Forest,
                               Gradient Boosting, XGBoost
    5. Select best model by CV accuracy
    6. Evaluate (confusion matrix, feature importance, ROC curve, model comparison)
    7. Save best model to models/
    8. Generate final Kaggle submission CSV -> submissions/best_submission.csv
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
    train_model,
    predict,
    cross_validate_model,
    save_model,
)
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

print("\n" + "="*60)
print("  Titanic ML -- Full Pipeline")
print("="*60)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("\n[1/8] Loading data...")
train_raw = load_data(TRAIN_PATH)
test_raw  = load_data(TEST_PATH)
display_info(train_raw, "Train (raw)")
display_info(test_raw,  "Test  (raw)")

test_ids = test_raw["PassengerId"].copy()

# ---------------------------------------------------------------------------
# 2+3. Feature engineering (needs Name) then preprocessing
#      Order matters: engineer() must run BEFORE clean() drops Name/Ticket/Cabin
# ---------------------------------------------------------------------------
print("\n[2/8] Feature engineering...")
train_fe = engineer(train_raw.copy())
test_fe  = engineer(test_raw.copy())

print("\n[3/8] Preprocessing...")
train_clean = clean(train_fe)
test_clean  = clean(test_fe)

print(f"  Train: {train_clean.shape}  |  Test: {test_clean.shape}")

# ---------------------------------------------------------------------------
# 4. Feature selection
# ---------------------------------------------------------------------------
print("\n[4/8] Selecting features...")
feature_cols = select_features(train_clean)
print(f"  {len(feature_cols)} features: {feature_cols}")

TARGET  = "Survived"
X_train = train_clean[feature_cols]
y_train = train_clean[TARGET]
X_test  = test_clean[feature_cols]

# ---------------------------------------------------------------------------
# 5. Train & cross-validate all models
# ---------------------------------------------------------------------------
print("\n[5/8] Cross-validating models (5-fold stratified)...")

MODEL_TYPES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

cv_results    = {}
fitted_models = {}

for mtype in MODEL_TYPES:
    label = mtype.replace("_", " ").title()
    print(f"\n  -- {label} --")
    model  = build_baseline_model(mtype)
    result = cross_validate_model(model, X_train, y_train, cv=5)
    cv_results[label] = result
    fitted_models[mtype] = train_model(build_baseline_model(mtype), X_train, y_train)

# ---------------------------------------------------------------------------
# 6. Select best model
# ---------------------------------------------------------------------------
print("\n[6/8] Selecting best model...")
best_label = max(cv_results, key=lambda k: cv_results[k]["mean"])
best_cv    = cv_results[best_label]["mean"]
print(f"  [BEST] {best_label}  (CV accuracy = {best_cv:.4f})")

key_map    = {m.replace("_", " ").title(): m for m in MODEL_TYPES}
best_model = fitted_models[key_map[best_label]]

# ---------------------------------------------------------------------------
# 7. Evaluate best model on training data (diagnostic)
# ---------------------------------------------------------------------------
print("\n[7/8] Evaluating best model on training data...")
y_train_pred = predict(best_model, X_train)
accuracy_score(y_train, y_train_pred)
print_classification_report(y_train, y_train_pred)

plot_confusion_matrix(y_train, y_train_pred, save=True)
plot_feature_importance(best_model, feature_cols, top_n=len(feature_cols), save=True)
compare_models(cv_results, save=True)

if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_train)[:, 1]
    roc_auc(y_train, y_prob, save=True)

save_model(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

# ---------------------------------------------------------------------------
# 8. Generate final Kaggle submission
# ---------------------------------------------------------------------------
print("\n[8/8] Generating submission...")
y_test_pred = predict(best_model, X_test)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived":    y_test_pred.astype(int),
})
save_submission(submission, SUBMISSION_PATH)

print("\n" + "="*60)
print("  [DONE] Pipeline complete!")
print(f"  Best model  -> {best_label}  |  CV = {best_cv:.4f}")
print(f"  Submission  -> {SUBMISSION_PATH}")
print("="*60 + "\n")
