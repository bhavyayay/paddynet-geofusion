import os
import sys
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import get_sh_config
CDSE_SENTINEL2_L2A = DataCollection.SENTINEL2_L2A.define_from(
    "CDSE_SENTINEL2_L2A", service_url="https://sh.dataspace.copernicus.eu"
)
# Fatehabad district, Haryana, India — approximate bounding box
# (min_lon, min_lat, max_lon, max_lat)
FATEHABAD_BBOX_COORDS = [75.40, 29.48, 75.45, 29.53]


RESOLUTION = 10  # meters per pixel, matching Sentinel-2 native resolution

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "dataMask"],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.dataMask];
}
"""


def download_sentinel2_image(
    bbox_coords=FATEHABAD_BBOX_COORDS,
    time_interval=("2022-09-01", "2022-09-15"),
    output_path="data/raw/fatehabad_sentinel2.tif",
):
    config = get_sh_config()

    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

    print(f"Requesting image for bbox: {bbox_coords}")
    print(f"Time interval: {time_interval}")
    print(f"Output image size (pixels): {size}")

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=CDSE_SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="leastCC",
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config,
    )

    print("Sending request to Copernicus Data Space... this may take a moment.")
    data = request.get_data()

    if not data:
        raise RuntimeError("No data returned. Check your date range and bbox.")

    image_array = data[0]  # shape: (height, width, 5) -> B02, B03, B04, B08, dataMask
    print("Received image with shape:", image_array.shape)

    height, width, _ = image_array.shape
    transform = from_bounds(*bbox_coords, width=width, height=height)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    band_names = ["B02_blue", "B03_green", "B04_red", "B08_nir", "data_mask"]

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=5,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(5):
            dst.write(image_array[:, :, i], i + 1)
            dst.set_band_description(i + 1, band_names[i])

    print(f"Saved image to: {output_path}")
    return output_path


if __name__ == "__main__":
    download_sentinel2_image()