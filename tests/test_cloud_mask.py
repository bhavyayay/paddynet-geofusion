import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.cloud_mask import apply_data_mask, compute_valid_pixel_fraction, is_scene_usable


def test_apply_data_mask_keeps_valid_pixels():
    band = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data_mask = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    result = apply_data_mask(band, data_mask, fill_value=0.0)

    assert np.array_equal(result, band)


def test_apply_data_mask_zeroes_invalid_pixels():
    band = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data_mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = apply_data_mask(band, data_mask, fill_value=0.0)
    expected = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    assert np.array_equal(result, expected)


def test_compute_valid_pixel_fraction_all_valid():
    data_mask = np.ones((10, 10), dtype=np.float32)
    fraction = compute_valid_pixel_fraction(data_mask)
    assert fraction == 1.0


def test_compute_valid_pixel_fraction_half_valid():
    data_mask = np.zeros((10, 10), dtype=np.float32)
    data_mask[:5, :] = 1.0
    fraction = compute_valid_pixel_fraction(data_mask)
    assert np.isclose(fraction, 0.5)


def test_is_scene_usable_above_threshold():
    data_mask = np.ones((10, 10), dtype=np.float32)
    assert is_scene_usable(data_mask, min_valid_fraction=0.7) is True


def test_is_scene_usable_below_threshold():
    data_mask = np.zeros((10, 10), dtype=np.float32)
    data_mask[:2, :] = 1.0  # only 20% valid
    assert is_scene_usable(data_mask, min_valid_fraction=0.7) is False