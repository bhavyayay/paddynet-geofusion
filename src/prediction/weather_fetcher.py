"""
weather_fetcher.py

Pulls daily weather data (rainfall, temperature, humidity) from NASA POWER
for each parcel's location and date, then merges it with Dev 1's parcel
classification output.

NASA POWER API docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
No API key required — it's a free, open endpoint.

Usage:
    python -m src.prediction.weather_fetcher
"""

import time
import requests
import pandas as pd

from src.prediction.dataset_loader import load_and_prepare

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameters we want: max/min temperature (C), precipitation (mm/day),
# relative humidity (%). Full parameter list:
# https://power.larc.nasa.gov/docs/services/api/temporal/daily/
WEATHER_PARAMETERS = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"


def fetch_weather_for_point(latitude: float, longitude: float, date: pd.Timestamp,
                              retries: int = 3, delay: float = 1.0) -> dict:
    """
    Fetch a single day's weather for one lat/long point from NASA POWER.

    Returns a dict like:
        {"t2m_max": 34.2, "t2m_min": 24.1, "precip_mm": 3.5, "rh2m": 68.2}

    Retries a few times on transient failures before giving up, since
    NASA POWER can occasionally be slow or briefly unavailable.
    """
    date_str = date.strftime("%Y%m%d")

    params = {
        "parameters": WEATHER_PARAMETERS,
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": date_str,
        "end": date_str,
        "format": "JSON",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(NASA_POWER_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            props = data["properties"]["parameter"]
            return {
                "t2m_max": props["T2M_MAX"][date_str],
                "t2m_min": props["T2M_MIN"][date_str],
                "precip_mm": props["PRECTOTCORR"][date_str],
                "rh2m": props["RH2M"][date_str],
            }
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{retries} failed for ({latitude}, {longitude}, {date_str}): {e}")
            if attempt < retries:
                time.sleep(delay)

    # If all retries failed, return NaNs rather than crashing the whole run —
    # one bad point shouldn't kill the entire dataset build.
    print(f"  Giving up on ({latitude}, {longitude}, {date_str}) after {retries} attempts. Filling with NaN.")
    return {"t2m_max": None, "t2m_min": None, "precip_mm": None, "rh2m": None}


def merge_weather_into_parcels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in the parcel dataframe, fetch weather for that
    parcel's exact location and date, and add it as new columns.
    """
    weather_rows = []

    print(f"Fetching weather for {len(df)} parcels from NASA POWER...")
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] parcel_id={row['parcel_id']} lat={row['latitude']} lon={row['longitude']} date={row['date'].date()}")
        weather = fetch_weather_for_point(row["latitude"], row["longitude"], row["date"])
        weather_rows.append(weather)

    weather_df = pd.DataFrame(weather_rows, index=df.index)
    merged = pd.concat([df, weather_df], axis=1)

    return merged


def build_merged_dataset(save_path: str = "data/processed/merged_features.parquet") -> pd.DataFrame:
    """
    Full flow: load paddy parcels -> fetch weather -> merge -> save checkpoint.

    Saves a local checkpoint so repeated runs don't have to re-call the
    NASA POWER API every time (it can be slow for many individual points).
    """
    paddy_df = load_and_prepare()
    merged = merge_weather_into_parcels(paddy_df)

    merged.to_parquet(save_path, index=False)
    print(f"\nSaved merged dataset to {save_path}")

    return merged


if __name__ == "__main__":
    merged = build_merged_dataset()
    print()
    print("Preview of merged parcel + weather data:")
    cols_to_show = ["parcel_id", "date", "ndvi", "t2m_max", "t2m_min", "precip_mm", "rh2m"]
    print(merged[cols_to_show].to_string())
