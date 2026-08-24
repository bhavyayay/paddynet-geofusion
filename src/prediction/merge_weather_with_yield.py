"""
merge_weather_with_yield.py

Attaches weather data (max/min temperature, rainfall, humidity) to the
district-level and state-level Haryana rice yield dataset built from
the Statistical Abstract (haryana_rice_yield_combined.csv).

Since these are annual/seasonal aggregates rather than exact survey
dates, we use a single representative date per row: September 15 of
the relevant year. This falls in the middle of the paddy vegetative
growth phase (per the original paper, NDVI/vegetation peaks in
September in this region), making it a reasonable single-day proxy
for "what the growing season's weather was like" that year.

IMPORTANT LIMITATION:
NASA POWER's daily data only goes back to 1981. The three earliest
state-level rows (1966-67, 1970-71, 1980-81) predate this and will
have NO weather data available -- they'll be kept in the output with
weather columns as NaN, clearly flagged, rather than silently dropped
or filled with made-up values.

Usage:
    python -m src.prediction.merge_weather_with_yield
"""

import time
import requests
import pandas as pd

INPUT_PATH = "data/ground_truth/haryana_rice_yield_combined.csv"
OUTPUT_PATH = "data/processed/haryana_yield_weather_dataset.parquet"

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
WEATHER_PARAMETERS = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"

# NASA POWER's daily record starts 1981-01-01. Anything before this
# cannot be fetched.
EARLIEST_AVAILABLE_YEAR = 1981

# Approximate centroid coordinates for Haryana state and each district.
# These are reasonable approximations for climate-matching purposes,
# not survey-grade precision -- adequate for pulling representative
# regional weather.
REGION_COORDS = {
    "Haryana": (29.06, 76.09),
    "Ambala": (30.38, 76.78),
    "Bhiwani": (28.79, 76.14),
    "Charkhi Dadri": (28.59, 76.27),
    "Faridabad": (28.41, 77.31),
    "Fatehabad": (29.51, 75.45),
    "Gurugram": (28.46, 77.03),
    "Hisar": (29.15, 75.72),
    "Jhajjar": (28.61, 76.66),
    "Jind": (29.32, 76.31),
    "Kaithal": (29.80, 76.40),
    "Karnal": (29.69, 76.99),
    "Kurukshetra": (29.97, 76.87),
    "Mahendragarh": (28.28, 76.15),
    "Nuh": (28.11, 77.00),
    "Palwal": (28.14, 77.33),
    "Panchkula": (30.69, 76.85),
    "Panipat": (29.39, 76.97),
    "Rewari": (28.20, 76.62),
    "Rohtak": (28.89, 76.61),
    "Sirsa": (29.53, 75.03),
    "Sonipat": (28.99, 77.02),
    "Yamunanagar": (30.13, 77.28),
}


def get_representative_date(year_clean: str) -> str:
    """
    Convert a fiscal-year string like '2010-11' into a single
    representative date (September 15 of the starting year), formatted
    as YYYYMMDD for the NASA POWER API.
    """
    start_year = int(year_clean.split("-")[0])
    return f"{start_year}0915", start_year


def fetch_weather(latitude: float, longitude: float, date_str: str,
                   retries: int = 3, delay: float = 1.0) -> dict:
    """
    Fetch a single day's weather for one lat/long point from NASA POWER.
    Returns None values if the request fails after retries.
    """
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
            data = response.json()
            props = data["properties"]["parameter"]
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


def merge_weather(df: pd.DataFrame) -> pd.DataFrame:
    weather_rows = []

    for idx, row in df.iterrows():
        region = row["region"]
        year_clean = row["year_clean"]
        date_str, start_year = get_representative_date(year_clean)

        if start_year < EARLIEST_AVAILABLE_YEAR:
            print(f"[{idx+1}/{len(df)}] {region} {year_clean}: before 1981, "
                  f"NASA POWER has no data -- filling weather as NaN.")
            weather_rows.append({"t2m_max": None, "t2m_min": None,
                                  "precip_mm": None, "rh2m": None})
            continue

        if region not in REGION_COORDS:
            print(f"[{idx+1}/{len(df)}] {region}: no coordinates on file -- skipping.")
            weather_rows.append({"t2m_max": None, "t2m_min": None,
                                  "precip_mm": None, "rh2m": None})
            continue

        lat, lon = REGION_COORDS[region]
        print(f"[{idx+1}/{len(df)}] Fetching weather for {region} on {date_str}...")
        weather = fetch_weather(lat, lon, date_str)
        weather_rows.append(weather)

    weather_df = pd.DataFrame(weather_rows, index=df.index)
    merged = pd.concat([df, weather_df], axis=1)
    return merged


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}\n")

    merged = merge_weather(df)

    n_missing = merged["t2m_max"].isna().sum()
    print(f"\nDone. {len(merged) - n_missing}/{len(merged)} rows have weather data.")
    if n_missing:
        print(f"{n_missing} rows are missing weather (pre-1981 or lookup failures) "
              f"-- these are kept in the dataset but flagged as NaN, not dropped "
              f"or fabricated.")

    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nPreview:")
    cols = ["level", "region", "year_clean", "yield_t_ha", "t2m_max", "t2m_min", "precip_mm", "rh2m"]
    print(merged[cols].to_string())
