# PaddyNet-GeoFusion — Developer 1 Module (Geo Data Pipeline & Segmentation)

## What this module does
1. Downloads Sentinel-2 satellite imagery for the Fatehabad, Haryana study region via Copernicus Data Space.
2. Computes vegetation indices (NDVI, NDWI, MSAVI) and applies cloud/no-data masking.
3. Extracts patches around labeled GPS ground-truth points.
4. Trains a patch-based CNN classifier (paddy / non_paddy / water).
5. Produces a final parcel-level prediction table for downstream modules.

## Handoff artifact for Developer 2 & Developer 3

**File:** `data/processed/parcel_predictions.parquet`

**Schema** (see `schemas/parcel_schema.py` -> `ParcelRecord`):

| Column | Type | Description |
|---|---|---|
| parcel_id | string | Unique identifier, e.g. "parcel_1" |
| latitude | float | GPS latitude |
| longitude | float | GPS longitude |
| date | string (YYYY-MM-DD) | Survey/observation date |
| ndvi | float | Vegetation index |
| ndwi | float | Water index |
| msavi | float | Soil-adjusted vegetation index |
| vv | float or null | Sentinel-1 VV backscatter (not yet integrated) |
| vh | float or null | Sentinel-1 VH backscatter (not yet integrated) |
| data_mask | float | 1.0 = valid pixel, 0.0 = cloud/invalid |
| class_label | string | "paddy", "non_paddy", or "water" |
| class_confidence | float | Model's confidence (0.0-1.0) in the predicted class |

## How to reproduce this pipeline end-to-end

\`\`\`bash
source venv/bin/activate
python -m src.ingestion.download_imagery
python -m src.preprocessing.process_raw_image
python -m src.preprocessing.extract_training_points
python -m src.segmentation.train
python -m src.segmentation.predict
\`\`\`

## Current known limitations (to be resolved with real data)
- Ground truth is currently 12 **placeholder** GPS points in `data/ground_truth/gps_points.csv`.
  Replace this file with real farmer-survey data (target: 100+ points per the original paper)
  before trusting any accuracy numbers.
- Sentinel-1 VV/VH backscatter columns are placeholders (null) -- integration is a follow-up task.
- Model is a patch-classifier CNN, not a full pixel-wise segmentation model. A U-Net architecture
  is already wired up in `src/segmentation/model.py` (`build_unet_segmentation_model`) and ready
  to train once full-field polygon masks are available instead of point labels.

## Setup
See `requirements.txt`. Requires a free Copernicus Data Space account and OAuth credentials
stored in a local `.env` file (not committed to Git) -- see `src/utils/config.py`.