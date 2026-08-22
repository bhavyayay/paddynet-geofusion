import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schemas.parcel_schema import validate_ground_truth_row, LABEL_TO_INT, INT_TO_LABEL


def test_validate_ground_truth_row_valid_paddy():
    row = {"latitude": 29.48, "longitude": 75.40, "label": "paddy"}
    assert validate_ground_truth_row(row) is True


def test_validate_ground_truth_row_invalid_label():
    row = {"latitude": 29.48, "longitude": 75.40, "label": "corn"}
    with pytest.raises(ValueError, match="Invalid label"):
        validate_ground_truth_row(row)


def test_validate_ground_truth_row_invalid_latitude():
    row = {"latitude": 200.0, "longitude": 75.40, "label": "paddy"}
    with pytest.raises(ValueError, match="Invalid latitude"):
        validate_ground_truth_row(row)


def test_validate_ground_truth_row_invalid_longitude():
    row = {"latitude": 29.48, "longitude": 500.0, "label": "paddy"}
    with pytest.raises(ValueError, match="Invalid longitude"):
        validate_ground_truth_row(row)


def test_label_int_mapping_is_consistent():
    for label, integer in LABEL_TO_INT.items():
        assert INT_TO_LABEL[integer] == label