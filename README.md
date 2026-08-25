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

# PaddyNet-GeoFusion — Developer 3 Module (Evaluation, API & Dashboard)

## What this module does
1. **Evaluates** Developer 2's trained models against real historical
   yield numbers using standard accuracy measurements
   (MAE, RMSE, R², MAPE) — `src/evaluation/`.
2. **Serves predictions** to other programs through a small REST API —
   `src/api/`.
3. **Visualizes** the whole project on a single dashboard page
   (predicted vs actual yield charts, accuracy metrics, parcel
   classification summary) — `src/dashboard/`.
4. **Integration-tests** the full chain: Dev 1's schema contract →
   Dev 2's models → Dev 3's evaluation/API/dashboard —
   `tests/test_integration.py` and `tests/test_evaluation_metrics.py`.

## Evaluation methodology (why these numbers are honest)

The evaluation reconstructs the **exact** train/test split Developer 2
used (same random seed 42, same 25% test size), so the "held-out test"
metrics are computed only on rows the models never saw during
training. Metrics are reported for both scopes so nobody mistakes
in-sample accuracy for real accuracy:

| Model | Scope | MAE (t/ha) | RMSE (t/ha) | R² | MAPE |
|---|---|---|---|---|---|
| Random Forest | held-out test | 0.462 | 0.694 | 0.157 | 14.5% |
| Random Forest | full dataset | 0.269 | 0.417 | 0.684 | 8.6% |
| CNN-LSTM | held-out test | 0.512 | 0.742 | 0.035 | 16.2% |
| CNN-LSTM | full dataset | 0.510 | 0.666 | 0.195 | 15.6% |

(CNN-LSTM numbers vary slightly between training runs; regenerate with
the commands below. The held-out Random Forest numbers independently
reproduce what Developer 2's train.py reported, which confirms the
split reconstruction is correct.)

**Caveat:** the held-out test set is only 12 rows. Treat all metrics
as directional until the team has more real data.

**Artifacts produced:**
- `data/processed/evaluation_report.json` — machine-readable metrics
  (also served live at the API's `/metrics` endpoint)
- `data/processed/evaluation_predictions.parquet` — per-row actual vs
  predicted comparison (feeds the dashboard charts)

## The API

FastAPI service exposing Dev 2's trained models. Unlike the library
interface (`src.prediction.predict`), it loads model files **once at
startup** instead of on every call.

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness + whether models are loaded |
| `POST /predict` | Yield prediction from area + weather features (validated inputs) |
| `GET /metrics` | Latest evaluation report |
| `GET /dashboard` | Human-friendly dashboard page |
| `GET /docs` | Interactive API documentation |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"area_000ha": 129.2, "t2m_max": 34.6, "t2m_min": 23.5, "precip_mm": 0.14, "rh2m": 70.4}'
# -> {"random_forest_t_ha": 3.708, "cnn_lstm_t_ha": ..., "recommended": 3.708, "recommended_model": "random_forest"}
```

**Security note:** the API has no authentication. It is for local
development and team demos only — do not expose it beyond localhost
without adding auth.

## The dashboard

A single self-contained HTML page (inline SVG charts, no JavaScript
libraries, no internet needed) showing:
- Predicted vs actual rice yield per Haryana district (2021-22)
- Haryana state yield trend over time, actual vs predicted
- The full accuracy metrics table with caveats
- Dev 1's parcel classification summary (when
  `parcel_predictions.parquet` exists; shows a clear note otherwise)

View it either at the API's `/dashboard` endpoint, or as a static file
at `reports/dashboard.html`.

## How to reproduce this module end-to-end

```bash
source venv/bin/activate
pip install -r requirements.txt

# Prerequisite: Dev 2's dataset + trained models
# (copy combined_training_dataset.parquet into data/processed/ if needed)
python -m src.prediction.train

# 1. Evaluate models vs real historical yields
python -m src.evaluation.evaluate_models

# 2. Build the static dashboard
python -m src.dashboard.build_dashboard      # writes reports/dashboard.html

# 3. Run the API (then open http://localhost:8000/dashboard)
uvicorn src.api.app:app --reload

# 4. Run all tests (Dev 1 + Dev 3, 37 tests)
python -m pytest tests/ -v
```

## Files in this module

- `src/evaluation/metrics.py` — MAE / RMSE / R² / MAPE as pure, tested functions
- `src/evaluation/evaluate_models.py` — full evaluation run + report artifacts
- `src/api/app.py` — FastAPI service (model caching, input validation)
- `src/dashboard/build_dashboard.py` — HTML dashboard generator
- `tests/test_evaluation_metrics.py` — 12 unit tests for the metrics
- `tests/test_integration.py` — 8 cross-module integration tests

---

# How the Whole Project Fits Together

```
 Sentinel-2 imagery          Govt yield statistics        Consumers
 (Copernicus)                + NASA POWER weather         (humans & programs)
      |                              |                          ^
      v                              v                          |
+---------------+   parquet   +---------------+   models   +---------------+
|  DEVELOPER 1  | ----------> |  DEVELOPER 2  | ---------> |  DEVELOPER 3  |
|  segmentation |             |  yield model  |            |  eval + API   |
|  (paddy/not)  |             |  (RF + LSTM)  |            |  + dashboard  |
+---------------+             +---------------+            +---------------+
 parcel_predictions            combined_training            evaluation_report.json
 .parquet                      _dataset.parquet             reports/dashboard.html
                               models/*.joblib|*.pt         REST API on :8000
```

**The contracts between us are files, not code.** Each developer's
output is a documented artifact (schema tables above) the next
developer reads. `tests/test_integration.py` guards these contracts —
if someone renames a column or changes the model interface, tests fail
before anyone's pipeline silently breaks.

**Onboarding path for someone new:**
1. Read this README top to bottom (one section per developer).
2. `pip install -r requirements.txt` in a venv.
3. Run the Developer 3 reproduction steps above — they exercise the
   whole chain and end with a dashboard you can look at.
4. Note the honest limitations flagged in every section: placeholder
   GPS points (Dev 1), 47 usable training rows (Dev 2), a 12-row
   held-out test set (Dev 3). The pipeline is real and tested;
   the accuracy numbers will only become trustworthy with more data.
