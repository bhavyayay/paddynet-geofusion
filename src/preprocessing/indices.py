import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    denom = nir + red
    ndvi = np.where(denom == 0, 0.0, (nir - red) / (denom + 1e-8))
    return ndvi.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index."""
    denom = green + nir
    ndwi = np.where(denom == 0, 0.0, (green - nir) / (denom + 1e-8))
    return ndwi.astype(np.float32)


def compute_msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Modified Soil Adjusted Vegetation Index."""
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
    msavi = np.nan_to_num(msavi, nan=0.0, posinf=0.0, neginf=0.0)
    return msavi.astype(np.float32)


def compute_all_indices(blue: np.ndarray, green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> dict:
    """Compute all three vegetation indices from raw bands and return as a dict."""
    return {
        "ndvi": compute_ndvi(nir, red),
        "ndwi": compute_ndwi(green, nir),
        "msavi": compute_msavi(nir, red),
    }