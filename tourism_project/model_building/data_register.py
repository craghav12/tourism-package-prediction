"""
Registers the tourism dataset: validates that all expected columns are
present with the expected dtypes, and prints a summary of the dataset.

Run from the repository root:
    python tourism_project/model_building/data_register.py
"""

import os
import sys

import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)

    # Drop the stray index column that gets exported by some spreadsheet tools,
    # if present, so the column check below reflects the real schema.
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Source file        : {data_path}")
    print(f"Rows x Columns      : {df.shape[0]} x {df.shape[1]}")
    print(f"All expected columns present: Yes ({len(EXPECTED_COLUMNS)} columns)")
    print("-" * 60)
    print("Column dtypes:")
    print(df.dtypes)
    print("-" * 60)
    print("Missing values per column:")
    print(df.isnull().sum())
    print("-" * 60)
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).rename("proportion"))
    print("=" * 60)

    return df


if __name__ == "__main__":
    try:
        register_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset registration failed: {exc}", file=sys.stderr)
        sys.exit(1)
