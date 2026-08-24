"""
combine_datasets.py

Combines two real, independently-sourced datasets into one final
training set:

  1. Haryana district-level + state-history data (already fetched,
     saved at data/processed/haryana_yield_weather_dataset.parquet)
     -- captures SPATIAL variation (how yield differs by district)
     within a single year (2021-22), plus a handful of historical
     Haryana state-level years.

  2. Multi-state, multi-year "Final Estimate" dataset (Bihar, Haryana,
     Punjab, Uttar Pradesh, Chandigarh -- 2021-22 through 2025-26)
     -- captures TEMPORAL + CROSS-STATE variation (how yield differs
     by year and by state), which the first dataset didn't have.

Together these give the model two genuinely different axes of real
variation to learn from, rather than just one.

Usage:
    python -m src.prediction.combine_datasets
"""

import time
import requests
import pandas as pd

EXISTING_DATASET_PATH = "data/processed/haryana_yield_weather_dataset.parquet"
NEW_RAW_CSV_PATH = "data/ground_truth/Final-Estimate-of-Area,-Production-&-Yield-for-Rice.csv"
OUTPUT_PATH = "data/processed/combined_training_dataset.parquet"

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
WEATHER_PARAMETERS = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"

# Approximate state centroids for the 5 states/UT in the new dataset.
STATE_COORDS = {
    "Bihar": (25.09, 85.31),
    "Haryana": (29.06, 76.09),
    "Punjab": (31.15, 75.34),
    "Uttar Pradesh": (26.85, 80.91),
    "Chandigarh": (30.73, 76.78),
}


def reshape_new_dataset(csv_path: str) -> pd.DataFrame:
    """
    Melts the wide (one-column-per-year) CSV into long format, keeping
    only the 'Total' season row per state/year to avoid double-counting
    (Kharif and Total are identical for states with no Rabi/Summer
    rice), and dropping zero-area rows (e.g. Chandigarh grows
    essentially no rice -- not a meaningful yield observation).
    """
    df = pd.read_csv(csv_path)
    years = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

    rows = []
    for _, r in df.iterrows():
        if r["Season"] != "Total":
            continue
        for y in years:
            area = r.get(f"Area-{y}")
            prod = r.get(f"Production-{y}")
            yld = r.get(f"Yield-{y}")
            if pd.notna(area) and pd.notna(prod) and pd.notna(yld) and area > 0:
                rows.append({
                    "level": "state",
                    "region": r["State"],
                    "year_clean": y,
                    "area_000ha": area * 100,       # lakh ha -> thousand ha
                    "production_000t": prod * 100,   # lakh t -> thousand t
                    "yield_t_ha": yld / 1000,        # kg/ha -> t/ha
                    "source": "Final Estimate of Area, Production, Yield for Rice (DES/data.gov.in)",
                })

    return pd.DataFrame(rows)


def fetch_weather(latitude: float, longitude: float, date_str: str,
                   retries: int = 3, delay: float = 1.0) -> dict:
    params = {
        "parameters": WEATHER_PARAMETERS,
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": date_str,
        "end": date_str,
        "format": "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(NASA_POWER_URL, params=params, timeout=30)
            response.raise_for_status()
            props = response.json()["properties"]["parameter"]
            return {
                "t2m_max": props["T2M_MAX"][date_str],
                "t2m_min": props["T2M_MIN"][date_str],
                "precip_mm": props["PRECTOTCORR"][date_str],
                "rh2m": props["RH2M"][date_str],
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                print(f"    Failed after {retries} attempts: {e}")
    return {"t2m_max": None, "t2m_min": None, "precip_mm": None, "rh2m": None}


def attach_weather_to_new_rows(new_df: pd.DataFrame) -> pd.DataFrame:
    weather_rows = []
    for idx, row in new_df.iterrows():
        region = row["region"]
        year_start = int(row["year_clean"].split("-")[0])
        date_str = f"{year_start}0915"

        if region not in STATE_COORDS:
            print(f"[{idx+1}/{len(new_df)}] {region}: no coordinates on file -- skipping.")
            weather_rows.append({"t2m_max": None, "t2m_min": None,
                                  "precip_mm": None, "rh2m": None})
            continue

        lat, lon = STATE_COORDS[region]
        print(f"[{idx+1}/{len(new_df)}] Fetching weather for {region} on {date_str}...")
        weather_rows.append(fetch_weather(lat, lon, date_str))

    weather_df = pd.DataFrame(weather_rows, index=new_df.index)
    return pd.concat([new_df, weather_df], axis=1)


if __name__ == "__main__":
    print("Loading existing Haryana district/state dataset...")
    existing = pd.read_parquet(EXISTING_DATASET_PATH)
    print(f"  {len(existing)} rows loaded.\n")

    print("Reshaping new multi-state multi-year dataset...")
    new_df = reshape_new_dataset(NEW_RAW_CSV_PATH)
    print(f"  {len(new_df)} usable rows after reshaping.\n")

    print("Fetching weather for new dataset rows...")
    new_with_weather = attach_weather_to_new_rows(new_df)

    # Align columns and combine
    common_cols = ["level", "region", "year_clean", "area_000ha", "yield_t_ha",
                    "t2m_max", "t2m_min", "precip_mm", "rh2m"]

    existing_aligned = existing[[c for c in common_cols if c in existing.columns]].copy()
    new_aligned = new_with_weather[[c for c in common_cols if c in new_with_weather.columns]].copy()

    combined = pd.concat([existing_aligned, new_aligned], ignore_index=True)

    print(f"\nCombined dataset: {len(combined)} total rows")
    n_complete = combined.dropna(subset=["t2m_max", "yield_t_ha", "area_000ha"]).shape[0]
    print(f"Rows with complete features + label: {n_complete}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nPreview:")
    print(combined.to_string())
