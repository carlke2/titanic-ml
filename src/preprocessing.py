"""
preprocessing.py
----------------
Data cleaning and missing value handling functions for the Titanic dataset.
"""

import pandas as pd
import numpy as np


def fill_age(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Age values using median grouped by Pclass and Sex.

    This is more accurate than a global median because age distribution
    varies significantly by passenger class and gender.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with Age column filled.
    """
    df = df.copy()
    age_medians = df.groupby(["Pclass", "Sex"])["Age"].median()

    def _fill(row):
        if pd.isnull(row["Age"]):
            return age_medians.loc[(row["Pclass"], row["Sex"])]
        return row["Age"]

    df["Age"] = df.apply(_fill, axis=1)
    return df


def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Embarked values with the mode (most common port = 'S').

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with Embarked column filled.
    """
    df = df.copy()
    mode_val = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(mode_val)
    return df


def fill_fare(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Fare values with the median fare for their Pclass.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with Fare column filled.
    """
    df = df.copy()
    fare_medians = df.groupby("Pclass")["Fare"].median()

    def _fill(row):
        if pd.isnull(row["Fare"]):
            return fare_medians.loc[row["Pclass"]]
        return row["Fare"]

    df["Fare"] = df.apply(_fill, axis=1)
    return df


def drop_unused_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not useful for modelling.

    Drops: Cabin (too many missing), Ticket (high cardinality noise),
    Name (title extracted separately in features.py).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with unused columns removed.
    """
    df = df.copy()
    cols_to_drop = ["Cabin", "Ticket", "Name"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)
    return df


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Sex column to numeric (male=0, female=1).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with Sex as integer.
    """
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df


def encode_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the Embarked column (C, Q, S → binary columns).

    Drops the first dummy to avoid multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with Embarked replaced by dummy columns.
    """
    df = df.copy()
    dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=["Embarked"], inplace=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline in the correct order.

    Parameters
    ----------
    df : pd.DataFrame  Raw Titanic DataFrame.

    Returns
    -------
    pd.DataFrame  Cleaned and encoded DataFrame.
    """
    df = fill_age(df)
    df = fill_embarked(df)
    df = fill_fare(df)
    df = encode_sex(df)
    df = encode_embarked(df)
    df = drop_unused_cols(df)
    return df
