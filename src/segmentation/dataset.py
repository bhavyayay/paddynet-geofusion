import os
import sys

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from schemas.parcel_schema import LABEL_TO_INT

PATCH_SIZE = 32  # pixels; a 32x32 patch centered on each GPS point


def extract_patch(band_stack: np.ndarray, center_row: int, center_col: int, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Extracts a (channels, patch_size, patch_size) patch centered at (center_row, center_col).
    band_stack shape: (channels, height, width)
    Pads with zeros if the patch would go outside image bounds.
    """
    channels, height, width = band_stack.shape
    half = patch_size // 2

    row_start = center_row - half
    row_end = center_row + half
    col_start = center_col - half
    col_end = center_col + half

    patch = np.zeros((channels, patch_size, patch_size), dtype=np.float32)

    src_row_start = max(row_start, 0)
    src_row_end = min(row_end, height)
    src_col_start = max(col_start, 0)
    src_col_end = min(col_end, width)

    dst_row_start = src_row_start - row_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - col_start
    dst_col_end = dst_col_start + (src_col_end - src_col_start)

    patch[:, dst_row_start:dst_row_end, dst_col_start:dst_col_end] = band_stack[
        :, src_row_start:src_row_end, src_col_start:src_col_end
    ]

    return patch


class PaddyPatchDataset(Dataset):
    """
    PyTorch Dataset that loads small image patches centered on labeled GPS points,
    for training/fine-tuning the segmentation model.
    """

    def __init__(
        self,
        points_parquet_path="data/processed/training_points.parquet",
        processed_tif_path="data/interim/fatehabad_processed.tif",
        patch_size=PATCH_SIZE,
        feature_bands=("blue", "green", "red", "nir", "ndvi", "ndwi", "msavi"),
    ):
        self.points_df = pd.read_parquet(points_parquet_path)
        self.patch_size = patch_size
        self.feature_bands = feature_bands

        with rasterio.open(processed_tif_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            all_band_names = [src.descriptions[i] for i in range(src.count)]
            self.band_indices = [all_band_names.index(b) + 1 for b in feature_bands]
            band_arrays = [src.read(i) for i in self.band_indices]
            self.band_stack = np.stack(band_arrays, axis=0).astype(np.float32)
            self.src_height = src.height
            self.src_width = src.width
            self.src_transform = src.transform

        print(f"Loaded {len(self.points_df)} points for patch dataset.")
        print(f"Using feature bands: {feature_bands}")
        print(f"Full image band stack shape: {self.band_stack.shape}")

    def __len__(self):
        return len(self.points_df)

    def __getitem__(self, idx):
        row = self.points_df.iloc[idx]
        lon, lat = row["longitude"], row["latitude"]

        with rasterio.open("data/interim/fatehabad_processed.tif") as src:
            pixel_row, pixel_col = src.index(lon, lat)

        patch = extract_patch(self.band_stack, pixel_row, pixel_col, self.patch_size)
        label_int = LABEL_TO_INT[row["label"]]

        patch_tensor = torch.from_numpy(patch).float()
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return patch_tensor, label_tensor


if __name__ == "__main__":
    dataset = PaddyPatchDataset()
    print(f"\nDataset size: {len(dataset)}")

    patch, label = dataset[0]
    print(f"First patch shape: {patch.shape}")
    print(f"First patch label: {label.item()}")
    print(f"First patch value range: min={patch.min().item():.4f}, max={patch.max().item():.4f}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_patches, batch_labels = next(iter(loader))
    print(f"\nBatch patches shape: {batch_patches.shape}")
    print(f"Batch labels: {batch_labels}")