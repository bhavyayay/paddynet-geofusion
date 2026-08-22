import numpy as np


def apply_data_mask(band: np.ndarray, data_mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Sets pixels to fill_value wherever the data_mask indicates invalid/no-data.
    data_mask: array where 1.0 = valid pixel, 0.0 = invalid/cloud/no-data pixel.
    """
    masked = np.where(data_mask > 0, band, fill_value)
    return masked.astype(np.float32)


def compute_valid_pixel_fraction(data_mask: np.ndarray) -> float:
    """Returns the fraction (0.0 to 1.0) of pixels that are valid (not cloud/no-data)."""
    total_pixels = data_mask.size
    valid_pixels = np.sum(data_mask > 0)
    return float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0


def is_scene_usable(data_mask: np.ndarray, min_valid_fraction: float = 0.7) -> bool:
    """
    Returns True if the scene has enough valid (non-cloud) pixels to be usable.
    Paper used a similar cloud coverage threshold (<10% cloud cover preferred).
    """
    valid_fraction = compute_valid_pixel_fraction(data_mask)
    return valid_fraction >= min_valid_fraction