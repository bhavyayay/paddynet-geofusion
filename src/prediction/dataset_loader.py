"""
dataset_loader.py

Loads the parcel-level classification output from Developer 1's pipeline
(data/processed/parcel_predictions.parquet) and prepares it for the
yield prediction module.

Usage:
    python -m src.prediction.dataset_loader
"""

import os
import pandas as pd

# Path to Dev 1's handoff file, relative to project root
PARCEL_PREDICTIONS_PATH = "data/processed/parcel_predictions.parquet"


def load_parcel_predictions(path: str = PARCEL_PREDICTIONS_PATH) -> pd.DataFrame:
    """
    Load Dev 1's parcel_predictions.parquet file.

    Raises a clear error if the file is missing, rather than a cryptic
    pandas/pyarrow traceback.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. "
            "Make sure Dev 1's parcel_predictions.parquet has been copied "
            "into data/processed/ before running this script."
        )

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def filter_paddy_parcels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows classified as 'paddy'. Yield prediction only makes
    sense for parcels that are actually growing paddy — non_paddy and
    water rows are not relevant to this module.
    """
    if "class_label" not in df.columns:
        raise ValueError(
            "Expected a 'class_label' column but it wasn't found. "
            f"Available columns: {df.columns.tolist()}"
        )

    before = len(df)
    paddy_df = df[df["class_label"] == "paddy"].copy()
    after = len(paddy_df)
    print(f"Filtered to paddy parcels: {after}/{before} rows kept")

    return paddy_df


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Confirm the columns we'll need downstream (for weather merging and
    modeling) are actually present. Fails loudly and early rather than
    letting a missing column cause a confusing error later in the
    pipeline.
    """
    required = ["parcel_id", "latitude", "longitude", "date", "ndvi", "ndwi", "msavi"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Check date is parseable — we'll need this for the weather API calls
    try:
        pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"Could not parse 'date' column as dates: {e}")

    print("All required columns present and valid.")


def load_and_prepare() -> pd.DataFrame:
    """
    Full loading pipeline: load -> validate -> filter to paddy.
    This is the function other scripts (like the weather merger) should
    import and call.
    """
    df = load_parcel_predictions()
    validate_required_columns(df)
    paddy_df = filter_paddy_parcels(df)

    # Ensure date is a proper datetime type for downstream use
    paddy_df["date"] = pd.to_datetime(paddy_df["date"])

    return paddy_df


if __name__ == "__main__":
    paddy_df = load_and_prepare()
    print()
    print("Preview of paddy parcels ready for weather merging:")
    print(paddy_df[["parcel_id", "latitude", "longitude", "date", "ndvi", "class_confidence"]].to_string())
