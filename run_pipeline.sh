#!/bin/bash
set -e

echo "=== PaddyNet-GeoFusion — Developer 1 Pipeline ==="
echo ""

echo "[1/6] Downloading satellite imagery..."
python -m src.ingestion.download_imagery

echo ""
echo "[2/6] Processing raw image (cloud mask + indices)..."
python -m src.preprocessing.process_raw_image

echo ""
echo "[3/6] Extracting training points..."
python -m src.preprocessing.extract_training_points

echo ""
echo "[4/6] Training patch classifier..."
python -m src.segmentation.train

echo ""
echo "[5/6] Running inference and saving predictions..."
python -m src.segmentation.predict

echo ""
echo "[6/6] Running test suite..."
python -m pytest tests/ -v

echo ""
echo "=== Pipeline completed successfully ==="