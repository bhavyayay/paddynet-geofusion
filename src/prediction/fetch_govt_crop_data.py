"""
fetch_govt_crop_data.py

Pulls district-wise, season-wise Rice production statistics for Haryana
from the data.gov.in API, filtering server-side where possible and
paginating through results rather than downloading the full 73,000+ row
dataset at once.

Dataset: "District-wise, season-wise crop production statistics from 1997"
Resource ID: 35be999b-0208-4354-b557-f6ca9a5355de
Source: https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997

SETUP REQUIRED:
    1. Get your free API key from data.gov.in (My Account -> API Keys)
    2. Paste it below, replacing "YOUR_API_KEY_HERE"

Usage:
    python -m src.prediction.fetch_govt_crop_data
"""

import time
import requests
import pandas as pd

# ---- CONFIG: replace with your real key before running ----
API_KEY = "579b464db66ec23bdd000001578f1c568ef84bb45d0b1b4e4b4aa399"
RESOURCE_ID = "35be999b-0208-4354-b557-f6ca9a5355de"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

PAGE_SIZE = 100  # smaller pages -- the server appears slow/overloaded,
                  # so smaller requests are more likely to complete
                  # before timing out than large 1000-row pulls
REQUEST_TIMEOUT = 120  # seconds to wait for a single page
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5  # wait longer after each consecutive failure
OUTPUT_PATH = "data/ground_truth/haryana_rice_yield_govt.csv"


def fetch_page(offset: int, limit: int = PAGE_SIZE) -> tuple:
    """
    Fetch a single page of results from the API, retrying on timeout
    since this particular government server appears to be slow/flaky
    rather than genuinely broken.

    We ask the API to filter by State and Crop server-side using the
    filters[...] query param syntax data.gov.in supports, so we don't
    have to download and discard 70k+ irrelevant rows locally.
    """
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "filters[State_Name]": "Haryana",
        "filters[Crop]": "Rice",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            records = data.get("records", [])
            total = data.get("total", None)
            return records, total

        except requests.exceptions.RequestException as e:
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  [offset={offset}] Attempt {attempt}/{MAX_RETRIES} failed "
                  f"({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  [offset={offset}] Giving up after {MAX_RETRIES} attempts. "
          f"Last error: {last_error}")
    return [], None


def fetch_all_haryana_rice() -> pd.DataFrame:
    """
    Loop through all pages until we've retrieved every matching row.
    """
    all_records = []
    offset = 0

    print("Fetching Haryana + Rice records from data.gov.in...")

    consecutive_empty_pages = 0

    while True:
        records, total = fetch_page(offset)

        if not records:
            consecutive_empty_pages += 1
            # One empty page might just mean we've reached the end.
            # Two in a row (after retries already happened inside
            # fetch_page) means something's genuinely wrong -- stop
            # rather than looping forever.
            if consecutive_empty_pages >= 2:
                print("  Two consecutive empty/failed pages -- stopping.")
                break
            offset += PAGE_SIZE
            continue

        consecutive_empty_pages = 0
        all_records.extend(records)
        print(f"  Retrieved {len(all_records)} rows so far"
              + (f" (of {total} total matching)" if total else ""))

        offset += PAGE_SIZE

        # Stop once we've retrieved everything the API reports
        if total is not None and offset >= int(total):
            break

        time.sleep(1)  # be polite to the server between requests

    df = pd.DataFrame(all_records)
    print(f"\nDone. Total rows retrieved: {len(df)}")
    return df


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Please replace API_KEY at the top of this file with your "
            "real data.gov.in API key before running."
        )

    df = fetch_all_haryana_rice()

    if df.empty:
        print("No records returned. Check that 'State_Name' and 'Crop' "
              "are the correct field names for this resource -- run "
              "fetch_page(0, limit=5) without filters first to inspect "
              "the raw field names if this happens.")
    else:
        print("\nColumns returned:", df.columns.tolist())
        print("\nPreview:")
        print(df.head(10).to_string())

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved to {OUTPUT_PATH}")
