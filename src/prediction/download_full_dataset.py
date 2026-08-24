"""
download_full_dataset.py

Downloads the FULL "District-wise, season-wise crop production statistics"
dataset from data.gov.in as CSV, using a streaming download so large
files don't time out the way a single big browser download can.

Then filters to Haryana + Rice locally using pandas -- much faster and
more reliable than asking the API to filter server-side, since this
particular government server appears to be slow/flaky under load.

SETUP REQUIRED:
    1. Get your free API key from data.gov.in (My Account -> API Keys)
    2. Paste it below, replacing "YOUR_API_KEY_HERE"

Usage:
    python -m src.prediction.download_full_dataset
"""

import os
import requests
import pandas as pd

# ---- CONFIG: replace with your real key before running ----
API_KEY = "579b464db66ec23bdd000001578f1c568ef84bb45d0b1b4e4b4aa399"
RESOURCE_ID = "35be999b-0208-4354-b557-f6ca9a5355de"

RAW_CSV_PATH = "data/raw/full_crop_production_dataset.csv"
FILTERED_OUTPUT_PATH = "data/ground_truth/haryana_rice_yield_govt.csv"


def download_full_csv(output_path: str = RAW_CSV_PATH) -> str:
    """
    Streams the full dataset as CSV directly to disk, without loading
    it all into memory at once. This avoids the timeout issues seen
    with both the browser download and the filtered/paginated API.
    """
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
    params = {
        "api-key": API_KEY,
        "format": "csv",
        "limit": 100000,  # ask for everything in one shot -- CSV format
                           # tends to be lighter/faster server-side than
                           # paginated JSON for large dumps
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Starting streamed download... this may take a few minutes for a large file.")
    with requests.get(url, params=params, stream=True, timeout=300) as response:
        response.raise_for_status()

        total_bytes = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes % (1024 * 1024) < 8192:  # roughly every 1MB
                        print(f"  Downloaded {total_bytes / (1024*1024):.1f} MB so far...")

    print(f"Download complete. Saved to {output_path} ({total_bytes / (1024*1024):.1f} MB)")
    return output_path


def filter_to_haryana_rice(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Loads the full CSV and filters locally to Haryana + Rice rows.
    Handles unknown/varying column names gracefully by printing them
    first, so we can adjust if the actual column names differ from
    what we expect.
    """
    print(f"\nLoading {csv_path} into pandas...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} total rows, {len(df.columns)} columns.")
    print("Columns:", df.columns.tolist())

    # Try to find the right column names -- data.gov.in resources
    # sometimes vary in casing/naming between datasets.
    state_col = next((c for c in df.columns if "state" in c.lower()), None)
    crop_col = next((c for c in df.columns if c.lower() == "crop"), None)

    if state_col is None or crop_col is None:
        raise ValueError(
            f"Could not auto-detect State/Crop columns. "
            f"Available columns: {df.columns.tolist()}. "
            f"Please check manually and adjust this script."
        )

    print(f"\nFiltering on column '{state_col}' == 'Haryana' and '{crop_col}' == 'Rice'...")
    filtered = df[
        (df[state_col].str.strip().str.lower() == "haryana")
        & (df[crop_col].str.strip().str.lower() == "rice")
    ].copy()

    print(f"Filtered down to {len(filtered)} rows.")
    return filtered


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Please replace API_KEY at the top of this file with your "
            "real data.gov.in API key before running."
        )

    download_full_csv()
    filtered = filter_to_haryana_rice()

    os.makedirs(os.path.dirname(FILTERED_OUTPUT_PATH), exist_ok=True)
    filtered.to_csv(FILTERED_OUTPUT_PATH, index=False)
    print(f"\nSaved filtered Haryana Rice data to {FILTERED_OUTPUT_PATH}")

    print("\nPreview:")
    print(filtered.head(10).to_string())
