import os
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

PREDICTIONS_PATH = "data/processed/parcel_predictions.parquet"
PROCESSED_TIF_PATH = "data/interim/fatehabad_processed.tif"
OUTPUT_PNG = "data/processed/predictions_map.png"

CLASS_COLORS = {
    "paddy": "#2ca02c",
    "non_paddy": "#d62728",
    "water": "#1f77b4",
}


def load_ndvi_band(tif_path=PROCESSED_TIF_PATH):
    with rasterio.open(tif_path) as src:
        band_names = [src.descriptions[i] for i in range(src.count)]
        ndvi_index = band_names.index("ndvi") + 1
        ndvi = src.read(ndvi_index)
        transform = src.transform
    return ndvi, transform


def latlon_to_pixel(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def visualize_predictions():
    df = pd.read_parquet(PREDICTIONS_PATH)
    print(f"Loaded {len(df)} predictions.")
    print(df[["parcel_id", "class_label", "class_confidence"]].to_string())

    ndvi, transform = load_ndvi_band()

    fig, ax = plt.subplots(figsize=(10, 10))
    ndvi_plot = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(ndvi_plot, ax=ax, fraction=0.046, pad=0.04, label="NDVI")

    for _, row in df.iterrows():
        pixel_row, pixel_col = latlon_to_pixel(row["longitude"], row["latitude"], transform)
        color = CLASS_COLORS.get(row["class_label"], "gray")
        ax.scatter(
            pixel_col, pixel_row,
            c=color, s=150, edgecolors="black", linewidths=1.5, zorder=5,
        )
        ax.annotate(
            row["parcel_id"].replace("parcel_", "#"),
            (pixel_col, pixel_row),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color="black",
        )

    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="Predicted Class")

    ax.set_title("Parcel Predictions over NDVI Map")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"\nSaved predictions map to: {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    visualize_predictions()