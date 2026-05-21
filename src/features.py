"""
features.py
-----------
Feature engineering functions for the Titanic dataset.
All functions accept a DataFrame, add new column(s), and return a copy.
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Title mapping — group rare titles into a single 'Rare' bucket
# ---------------------------------------------------------------------------
_TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",   # French equivalent of Miss
    "Ms": "Miss",
    "Lady": "Rare",
    "Sir": "Rare",
    "Mme": "Mrs",     # French equivalent of Mrs
    "Capt": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
}

_TITLE_ENCODE = {
    "Mr": 1,
    "Miss": 2,
    "Mrs": 3,
    "Master": 4,
    "Rare": 5,
}


def create_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract passenger title from the Name column and encode it.

    Steps:
      1. Regex-extract the title word (e.g. 'Mr', 'Mrs', 'Miss')
      2. Map uncommon titles to 'Rare'
      3. Label-encode to an integer column 'Title'

    Parameters
    ----------
    df : pd.DataFrame  Must contain a 'Name' column.

    Returns
    -------
    pd.DataFrame with new 'Title' (int) column.
    """
    df = df.copy()
    # Extract title from name string, e.g. "Braund, Mr. Owen" → "Mr"
    df["Title"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].map(_TITLE_MAP).fillna("Rare")
    df["Title"] = df["Title"].map(_TITLE_ENCODE).fillna(5).astype(int)
    return df


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize = SibSp + Parch + 1 (include the passenger themselves).

    Parameters
    ----------
    df : pd.DataFrame  Must contain 'SibSp' and 'Parch' columns.

    Returns
    -------
    pd.DataFrame with new 'FamilySize' column.
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Create IsAlone = 1 if passenger is travelling alone (FamilySize == 1).

    Requires create_family_size() to have been run first.

    Parameters
    ----------
    df : pd.DataFrame  Must contain 'FamilySize' column.

    Returns
    -------
    pd.DataFrame with new 'IsAlone' (int) column.
    """
    df = df.copy()
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def create_fare_band(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Bin Fare into quantile-based bands and encode as integer (0-n_bins-1).

    Parameters
    ----------
    df : pd.DataFrame  Must contain 'Fare' column.
    n_bins : int  Number of quantile bands (default 4).

    Returns
    -------
    pd.DataFrame with new 'FareBand' (int) column.
    """
    df = df.copy()
    fare = df["Fare"].fillna(df["Fare"].median())
    df["FareBand"] = pd.qcut(fare, q=n_bins, labels=False, duplicates="drop")
    df["FareBand"] = df["FareBand"].fillna(0).astype(int)
    return df


def create_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bin Age into 5 equal-width groups and encode as integer (0-4).

    NaN Age values are temporarily filled with the column median for binning.
    Bands: 0-16, 16-32, 32-48, 48-64, 64-80.

    Parameters
    ----------
    df : pd.DataFrame  Must contain 'Age' column.

    Returns
    -------
    pd.DataFrame with new 'AgeBand' (int) column.
    """
    df = df.copy()
    age = df["Age"].fillna(df["Age"].median())
    df["AgeBand"] = pd.cut(
        age,
        bins=[0, 16, 32, 48, 64, 80],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    )
    df["AgeBand"] = df["AgeBand"].astype(int)
    return df


def select_features(df: pd.DataFrame) -> list:
    """Return the final list of feature column names for modelling.

    Parameters
    ----------
    df : pd.DataFrame  Fully preprocessed and feature-engineered DataFrame.

    Returns
    -------
    list of str — column names to use as model inputs.
    """
    candidate_features = [
        "Pclass",
        "Sex",
        "AgeBand",
        "FareBand",
        "FamilySize",
        "IsAlone",
        "Title",
        "Embarked_Q",
        "Embarked_S",
        "SibSp",
        "Parch",
    ]
    # Only return columns that actually exist in the DataFrame
    return [f for f in candidate_features if f in df.columns]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in the correct order.

    Parameters
    ----------
    df : pd.DataFrame  Cleaned Titanic DataFrame (from preprocessing.clean).
                       Must still have 'Name' column before title extraction.

    Returns
    -------
    pd.DataFrame with all engineered features added.
    """
    df = create_title(df)
    df = create_family_size(df)
    df = create_is_alone(df)
    df = create_fare_band(df)
    df = create_age_band(df)
    return df
