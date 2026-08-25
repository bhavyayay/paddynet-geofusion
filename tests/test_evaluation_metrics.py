import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.metrics import mae, rmse, r2, mape, compute_all


def test_mae_perfect_prediction_is_zero():
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == 0.0


def test_mae_known_value():
    # errors: 0.5, 0.5 -> mean 0.5
    assert np.isclose(mae([1.0, 2.0], [1.5, 1.5]), 0.5, atol=1e-6)


def test_rmse_known_value():
    # errors: 1, -1 -> sqrt(mean(1, 1)) = 1
    assert np.isclose(rmse([2.0, 2.0], [3.0, 1.0]), 1.0, atol=1e-6)


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [0.0, 0.0, 0.0, 4.0]  # one big miss
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_r2_perfect_prediction_is_one():
    y = [1.0, 2.0, 3.0, 4.0]
    assert np.isclose(r2(y, y), 1.0, atol=1e-6)


def test_r2_mean_prediction_is_zero():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.5, 2.5, 2.5, 2.5]  # always predict the mean
    assert np.isclose(r2(y_true, y_pred), 0.0, atol=1e-6)


def test_r2_constant_true_values_returns_zero_not_nan():
    assert r2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_mape_known_value():
    # |4-5|/4 = 25%, |2-2|/2 = 0% -> mean 12.5%
    assert np.isclose(mape([4.0, 2.0], [5.0, 2.0]), 12.5, atol=1e-6)


def test_mape_skips_zero_true_values():
    # The row with y_true == 0 must be excluded, not produce inf
    result = mape([0.0, 2.0], [1.0, 2.5])
    assert np.isfinite(result)
    assert np.isclose(result, 25.0, atol=1e-6)


def test_compute_all_returns_expected_keys():
    result = compute_all([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert set(result.keys()) == {"mae_t_ha", "rmse_t_ha", "r2", "mape_pct", "n_samples"}
    assert result["n_samples"] == 3


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


def test_empty_arrays_raise():
    with pytest.raises(ValueError):
        rmse([], [])
