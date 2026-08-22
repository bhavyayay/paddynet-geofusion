import os
import sys

import pandas as pd
import rasterio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from schemas.parcel_schema import validate_ground_truth_row, LABEL_TO_INT


def load_ground_truth(csv_path="data/ground_truth/gps_points.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} ground truth points.")

    for _, row in df.iterrows():
        validate_ground_truth_row(row.to_dict())

    print("All rows passed schema validation.")
    return df


def extract_pixel_values(df: pd.DataFrame, processed_tif_path="data/interim/fatehabad_processed.tif") -> pd.DataFrame:
    with rasterio.open(processed_tif_path) as src:
        band_names = [src.descriptions[i] for i in range(src.count)]
        print("Bands available in processed image:", band_names)

        records = []
        skipped = 0

        for _, row in df.iterrows():
            lon, lat = row["longitude"], row["latitude"]
            try:
                row_idx, col_idx = src.index(lon, lat)
            except Exception:
                skipped += 1
                continue

            if row_idx < 0 or row_idx >= src.height or col_idx < 0 or col_idx >= src.width:
                skipped += 1
                continue

            pixel_values = {}
            for band_i, name in enumerate(band_names, start=1):
                value = src.read(band_i)[row_idx, col_idx]
                pixel_values[name] = float(value)

            record = {
                "point_id": row["point_id"],
                "latitude": lat,
                "longitude": lon,
                "label": row["label"],
                "label_int": LABEL_TO_INT[row["label"]],
                "survey_date": row["survey_date"],
                **pixel_values,
            }
            records.append(record)

        if skipped > 0:
            print(f"Warning: {skipped} points fell outside the image bounds and were skipped.")

    result_df = pd.DataFrame(records)
    print(f"Successfully extracted pixel values for {len(result_df)} points.")
    return result_df


def main():
    df = load_ground_truth()
    result_df = extract_pixel_values(df)

    output_path = "data/processed/training_points.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    print(f"Saved training points table to: {output_path}")

    print("\nPreview of extracted data:")
    print(result_df.head(10).to_string())

    print("\nLabel distribution:")
    print(result_df["label"].value_counts())


if __name__ == "__main__":
    main()