"""
helpers.py
----------
General helper and utility functions for the Titanic ML project.
"""

import os
import random
import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)
    return df


def save_submission(df: pd.DataFrame, filepath: str) -> None:
    """Save a Kaggle submission CSV with PassengerId and Survived columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'PassengerId' and 'Survived' columns.
    filepath : str
        Output file path (e.g., 'submissions/my_submission.csv').
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    submission = df[["PassengerId", "Survived"]].copy()
    submission["Survived"] = submission["Survived"].astype(int)
    submission.to_csv(filepath, index=False)
    print(f"[OK] Submission saved -> {filepath}  ({len(submission)} rows)")


def display_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print shape, dtypes, missing value counts, and first few rows.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    name : str
        Label to display in the header.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Dtypes:\n{df.dtypes.to_string()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\n  Missing values:\n{missing.to_string()}")
    else:
        print("\n  ✅ No missing values")
    print(f"\n  First 5 rows:\n{df.head().to_string()}")
    print(f"{'='*60}\n")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] Random seed set to {seed}")
