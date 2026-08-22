import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.segmentation.dataset import PaddyPatchDataset
from src.segmentation.model import PatchClassifierCNN
from schemas.parcel_schema import INT_TO_LABEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/patch_classifier_final.pt"
OUTPUT_PATH = "data/processed/parcel_predictions.parquet"


def run_inference():
    dataset = PaddyPatchDataset()

    model = PatchClassifierCNN(in_channels=7, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"\nLoaded model from: {MODEL_PATH}")
    print(f"Running inference on {len(dataset)} points...\n")

    records = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            patch, true_label = dataset[idx]
            row = dataset.points_df.iloc[idx]

            patch_input = patch.unsqueeze(0).to(DEVICE)
            logits = model(patch_input)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()

            record = {
                "parcel_id": f"parcel_{row['point_id']}",
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "date": row["survey_date"],
                "ndvi": row["ndvi"],
                "ndwi": row["ndwi"],
                "msavi": row["msavi"],
                "vv": None,   # placeholder -- Sentinel-1 backscatter not yet integrated
                "vh": None,   # placeholder -- Sentinel-1 backscatter not yet integrated
                "data_mask": row["data_mask"],
                "class_label": INT_TO_LABEL[predicted_class],
                "class_confidence": confidence,
            }
            records.append(record)

            print(
                f"{record['parcel_id']}: predicted={record['class_label']} "
                f"(confidence={confidence:.2%}), true_label={INT_TO_LABEL[true_label.item()]}"
            )

    output_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved final parcel predictions table to: {OUTPUT_PATH}")
    print(f"Total records: {len(output_df)}")
    print("\nSchema of output table:")
    print(output_df.dtypes)

    return output_df


if __name__ == "__main__":
    run_inference()