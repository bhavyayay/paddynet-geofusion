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

**Option A — one command:**

\`\`\`bash
source venv/bin/activate
./run_pipeline.sh
\`\`\`

**Option B — step by step:**

\`\`\`bash
source venv/bin/activate
python -m src.ingestion.download_imagery
python -m src.preprocessing.process_raw_image
python -m src.preprocessing.extract_training_points
python -m src.segmentation.train
python -m src.segmentation.predict
python -m pytest tests/ -v
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

# PaddyNet-GeoFusion — Developer 2 Module (Yield Prediction)

## What this module does
1. Loads Developer 1's parcel-level paddy classification output.
2. Sources real historical/regional rice yield data from official
   government publications (since per-parcel yield ground truth
   doesn't exist yet -- see "Data sources" below).
3. Pulls matching weather data (temperature, rainfall, humidity) for
   each yield data point via the NASA POWER API.
4. Trains and compares two models -- a Random Forest baseline and a
   small CNN-LSTM (matching the architectural style of the reference
   paper) -- to predict rice yield (tonnes/hectare) from area and
   weather features.

## Data sources

Per-parcel yield labels don't exist for this project (Dev 1's 12
classified parcels have no matching real yield measurements). Instead,
this module sources real yield data at the district/state level from
two official government publications:

| Source | Rows | Coverage |
|---|---|---|
| State Statistical Abstract of Haryana 2022-23, Table 14.7 | 30 | Haryana state-level, 1966-2021 (9 years) + all 21 rice-growing Haryana districts, 2021-22 |
| "Final Estimate of Area, Production & Yield for Rice" (https://upag.gov.in/dash-reports/fiveyearapy) | 20 | Bihar, Haryana, Punjab, Uttar Pradesh -- state-level, 2021-22 to 2025-26 |

Combined and de-duplicated: **50 total rows, 47 with complete weather
+ yield features** (3 pre-1981 rows lack weather because NASA POWER's
daily record only starts in 1981).

Weather (max/min temperature, rainfall, humidity) is fetched from
[NASA POWER](https://power.larc.nasa.gov/) for each row's region
centroid, using September 15 of the relevant year as a representative
date (mid-way through the paddy vegetative growth phase per the
original reference paper's findings).

**Handoff artifact:** `data/processed/combined_training_dataset.parquet`

## Handoff artifact schema

| Column | Type | Description |
|---|---|---|
| level | string | "state" or "district" |
| region | string | Name of the state or district |
| year_clean | string | Agricultural year, e.g. "2021-22" |
| area_000ha | float | Rice-cultivated area, thousand hectares |
| yield_t_ha | float | Rice yield, tonnes/hectare (**target variable**) |
| t2m_max | float | Max daily temperature (°C), Sept 15 of that year |
| t2m_min | float | Min daily temperature (°C), Sept 15 of that year |
| precip_mm | float | Precipitation (mm), Sept 15 of that year |
| rh2m | float | Relative humidity (%), Sept 15 of that year |

## Models and results

Two models were trained and compared on a 35-row train / 12-row test
split:

| Model | MAE (t/ha) | RMSE (t/ha) | R² |
|---|---|---|---|
| Random Forest | 0.46 | 0.69 | **0.157** |
| Small CNN-LSTM | 0.51 | 0.76 | **-0.006** |

**Random Forest genuinely outperforms the CNN-LSTM here.** This is an
expected, honest result, not a bug: with only 35 training rows, deep
learning models don't have enough data to find real patterns, while
tree-based methods like Random Forest handle small tabular datasets
far better. The Random Forest's R² of 0.157 is modest but real --
it explains roughly 16% of yield variance, better than guessing the
mean every time. The CNN-LSTM's near-zero R² means it's essentially
predicting close to the average regardless of input.

Random Forest's feature importances (most to least predictive):
`rh2m` (humidity) > `area_000ha` > `precip_mm` > `t2m_max` > `t2m_min`.
Humidity being the top predictor is a plausible real-world signal, as
it closely tracks monsoon strength, which drives paddy yield.

## Why the CNN-LSTM is small (601 parameters, not 281,489)

The reference paper's CNN-LSTM used 281,489 trainable parameters,
trained on dense multi-decade historical records. This project
currently has 35-47 real rows. A model anywhere near the reference
paper's size would simply memorize this small dataset rather than
learn anything generalizable. The architecture here (`model.py`)
intentionally uses very few filters/units to have any realistic
chance of generalizing at this data size. It should be read as a
**pipeline proof-of-concept** -- it demonstrates the full
architecture and training loop work correctly end-to-end -- not
as a production-ready predictor.

## Known limitations (to be resolved with more data)

- **Small sample size.** 35-47 rows is small for any model, especially
  deep learning. Both models' results should be read as directional,
  not precise.
- **No true per-parcel yield labels.** Yield data is at
  district/state granularity, not matched to Dev 1's individual
  classified parcels. A parcel-level model would need real per-field
  yield measurements, which don't currently exist for this project.
- **Single-day weather proxy.** Each row uses one representative day
  (Sept 15) rather than a full growing-season time series, which
  limits how much temporal pattern the LSTM layer can actually learn.
- **Bulk government dataset inaccessible.** The much larger
  "District-wise, season-wise crop production statistics from 1997"
  dataset on data.gov.in (73,171 rows, would give far more training
  data) could not be downloaded -- the server returned timeouts and
  502 errors across multiple access methods (paginated API, bulk CSV
  API, browser download, direct file link) during development. Worth
  retrying later; if it becomes accessible, it would substantially
  improve both models' results without requiring any code changes --
  just re-running `combine_datasets.py` with the new source added.

## How to reproduce this module end-to-end

```bash
venv\Scripts\activate
python -m src.prediction.dataset_loader
python -m src.prediction.weather_fetcher
python -m src.prediction.merge_weather_with_yield
python -m src.prediction.combine_datasets
python -m src.prediction.train
```

## Files in this module

- `dataset_loader.py` -- loads and validates Dev 1's parcel predictions
- 'download_full_dataset.py'
- `weather_fetcher.py` -- fetches NASA POWER weather for Dev 1's parcels
- `merge_weather_with_yield.py` -- attaches weather to the Statistical Abstract dataset
- `combine_datasets.py` -- merges both real data sources into the final training set
- `model.py` -- small CNN-LSTM architecture (PyTorch)
- `train.py` -- trains and compares Random Forest vs. CNN-LSTM

## Setup

Requires `scikit-learn` in addition to the packages in the root
`requirements.txt` (not yet added there):

```bash
pip install scikit-learn
```
