"""
Loads the tourism dataset from the repository data folder, cleans it,
and splits it into train/test sets saved locally as CSV files.

Run from the repository root:
    python tourism_project/model_building/prep.py
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET_COLUMN = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop the stray pandas index column exported alongside the CSV, and
    # CustomerID, which is a unique identifier with no predictive value.
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols + ["CustomerID"], errors="ignore")

    # Fix inconsistent category labels found in the raw data.
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows and rows missing the target label, if any.
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Cleaned dataset shape : {df.shape}")
    print(f"Xtrain shape          : {Xtrain.shape}")
    print(f"Xtest shape           : {Xtest.shape}")
    print(f"Train target balance  :\n{ytrain.value_counts(normalize=True)}")
    print(f"Test target balance   :\n{ytest.value_counts(normalize=True)}")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
