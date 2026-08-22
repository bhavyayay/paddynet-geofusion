from dataclasses import dataclass
from typing import Optional


@dataclass
class GroundTruthPoint:
    """Schema for a single labeled GPS ground-truth point."""
    point_id: int
    latitude: float
    longitude: float
    label: str  # one of: "paddy", "non_paddy", "water"
    survey_date: str  # format: YYYY-MM-DD


VALID_LABELS = {"paddy", "non_paddy", "water"}

LABEL_TO_INT = {
    "non_paddy": 0,
    "paddy": 1,
    "water": 2,
}

INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class ParcelRecord:
    """
    Schema for the final output table Developer 1 produces.
    This is the contract consumed by Developer 2 (model training)
    and Developer 3 (evaluation/API).
    """
    parcel_id: str
    latitude: float
    longitude: float
    date: str
    ndvi: float
    ndwi: float
    msavi: float
    vv: Optional[float]
    vh: Optional[float]
    data_mask: float
    class_label: str
    class_confidence: float


def validate_ground_truth_row(row: dict) -> bool:
    """Validates a single row of ground truth data against the schema rules."""
    if row.get("label") not in VALID_LABELS:
        raise ValueError(f"Invalid label '{row.get('label')}'. Must be one of {VALID_LABELS}.")
    lat = float(row.get("latitude"))
    lon = float(row.get("longitude"))
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}")
    return True