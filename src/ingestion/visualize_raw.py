import numpy as np
import rasterio
import matplotlib.pyplot as plt


def visualize_rgb(tif_path="data/raw/fatehabad_sentinel2.tif", output_png="data/raw/preview_rgb.png"):
    with rasterio.open(tif_path) as src:
        print("Number of bands:", src.count)
        for i in range(1, src.count + 1):
            print(f"Band {i} description:", src.descriptions[i - 1])

        blue = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        data_mask = src.read(5).astype(np.float32)

    print("Red band stats -> min:", red.min(), "max:", red.max(), "mean:", red.mean())
    print("NIR band stats -> min:", nir.min(), "max:", nir.max(), "mean:", nir.mean())
    print("Data mask unique values:", np.unique(data_mask))

    def normalize(band, low_percentile=2, high_percentile=98):
        low = np.percentile(band, low_percentile)
        high = np.percentile(band, high_percentile)
        band_clipped = np.clip(band, low, high)
        band_norm = (band_clipped - low) / (high - low + 1e-8)
        return band_norm

    rgb = np.dstack([normalize(red), normalize(green), normalize(blue)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(rgb)
    axes[0].set_title("True Color (RGB)")
    axes[0].axis("off")

    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi_plot = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI (Vegetation Index)")
    axes[1].axis("off")
    fig.colorbar(ndvi_plot, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Saved preview image to: {output_png}")
    plt.show()


if __name__ == "__main__":
    visualize_rgb()