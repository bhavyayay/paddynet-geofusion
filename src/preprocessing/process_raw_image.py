import os
import sys

import numpy as np
import rasterio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocessing.indices import compute_all_indices
from src.preprocessing.cloud_mask import apply_data_mask, compute_valid_pixel_fraction, is_scene_usable


def process_image(
    input_path="data/raw/fatehabad_sentinel2.tif",
    output_path="data/interim/fatehabad_processed.tif",
):
    with rasterio.open(input_path) as src:
        profile = src.profile
        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        data_mask = src.read(5).astype(np.float32)

    valid_fraction = compute_valid_pixel_fraction(data_mask)
    usable = is_scene_usable(data_mask, min_valid_fraction=0.7)

    print(f"Valid pixel fraction: {valid_fraction:.4f}")
    print(f"Scene usable (>= 70% valid pixels): {usable}")

    if not usable:
        raise ValueError(
            f"Scene has too many invalid/cloud pixels ({valid_fraction:.2%} valid). "
            "Consider choosing a different date range."
        )

    blue_m = apply_data_mask(blue, data_mask)
    green_m = apply_data_mask(green, data_mask)
    red_m = apply_data_mask(red, data_mask)
    nir_m = apply_data_mask(nir, data_mask)

    indices = compute_all_indices(blue_m, green_m, red_m, nir_m)
    ndvi = indices["ndvi"]
    ndwi = indices["ndwi"]
    msavi = indices["msavi"]

    print("NDVI stats -> min:", np.nanmin(ndvi), "max:", np.nanmax(ndvi), "mean:", np.nanmean(ndvi))
    print("NDWI stats -> min:", np.nanmin(ndwi), "max:", np.nanmax(ndwi), "mean:", np.nanmean(ndwi))
    print("MSAVI stats -> min:", np.nanmin(msavi), "max:", np.nanmax(msavi), "mean:", np.nanmean(msavi))

    band_stack = np.stack([blue_m, green_m, red_m, nir_m, ndvi, ndwi, msavi, data_mask], axis=0)
    band_names = ["blue", "green", "red", "nir", "ndvi", "ndwi", "msavi", "data_mask"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(count=8, dtype="float32")

    with rasterio.open(output_path, "w", **out_profile) as dst:
        for i, name in enumerate(band_names):
            band_data = band_stack[i]
            band_data = np.nan_to_num(band_data, nan=0.0)
            dst.write(band_data, i + 1)
            dst.set_band_description(i + 1, name)

    print(f"Saved processed image with 8 bands to: {output_path}")
    return output_path


if __name__ == "__main__":
    process_image()