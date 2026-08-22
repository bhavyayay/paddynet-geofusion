import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.indices import compute_ndvi, compute_ndwi, compute_msavi, compute_all_indices


def test_compute_ndvi_basic_values():
    nir = np.array([[0.5, 0.8]], dtype=np.float32)
    red = np.array([[0.2, 0.1]], dtype=np.float32)
    ndvi = compute_ndvi(nir, red)

    expected_0 = (0.5 - 0.2) / (0.5 + 0.2)
    expected_1 = (0.8 - 0.1) / (0.8 + 0.1)

    assert ndvi.shape == (1, 2)
    assert np.isclose(ndvi[0, 0], expected_0, atol=1e-4)
    assert np.isclose(ndvi[0, 1], expected_1, atol=1e-4)


def test_compute_ndvi_range_is_valid():
    nir = np.random.uniform(0, 1, size=(10, 10)).astype(np.float32)
    red = np.random.uniform(0, 1, size=(10, 10)).astype(np.float32)
    ndvi = compute_ndvi(nir, red)

    assert ndvi.min() >= -1.0001
    assert ndvi.max() <= 1.0001


def test_compute_ndvi_handles_zero_denominator():
    nir = np.array([[0.0]], dtype=np.float32)
    red = np.array([[0.0]], dtype=np.float32)
    ndvi = compute_ndvi(nir, red)

    assert ndvi[0, 0] == 0.0
    assert not np.isnan(ndvi[0, 0])


def test_compute_ndwi_basic_values():
    green = np.array([[0.4]], dtype=np.float32)
    nir = np.array([[0.3]], dtype=np.float32)
    ndwi = compute_ndwi(green, nir)

    expected = (0.4 - 0.3) / (0.4 + 0.3)
    assert np.isclose(ndwi[0, 0], expected, atol=1e-4)


def test_compute_msavi_no_nans_on_valid_input():
    nir = np.random.uniform(0.1, 0.9, size=(10, 10)).astype(np.float32)
    red = np.random.uniform(0.1, 0.9, size=(10, 10)).astype(np.float32)
    msavi = compute_msavi(nir, red)

    assert not np.isnan(msavi).any()
    assert not np.isinf(msavi).any()


def test_compute_all_indices_returns_correct_keys():
    blue = np.random.uniform(0, 1, size=(5, 5)).astype(np.float32)
    green = np.random.uniform(0, 1, size=(5, 5)).astype(np.float32)
    red = np.random.uniform(0, 1, size=(5, 5)).astype(np.float32)
    nir = np.random.uniform(0, 1, size=(5, 5)).astype(np.float32)

    result = compute_all_indices(blue, green, red, nir)

    assert set(result.keys()) == {"ndvi", "ndwi", "msavi"}
    assert result["ndvi"].shape == (5, 5)
    assert result["ndwi"].shape == (5, 5)
    assert result["msavi"].shape == (5, 5)