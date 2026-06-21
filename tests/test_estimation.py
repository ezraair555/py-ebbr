"""Tests for pyebbr.estimation module (added 2026-06-21, Lane 4 of grade recovery)."""
import numpy as np
import pandas as pd
import pytest
from pyebbr import fit_beta_prior, add_ebb_estimate, posterior_parameters


@pytest.fixture
def fitted_prior():
    return fit_beta_prior(successes=[3, 8, 40], totals=[10, 40, 120])


def test_posterior_parameters_returns_alpha_beta(fitted_prior):
    """posterior_parameters should return alpha and beta arrays."""
    a, b = posterior_parameters(successes=[3, 8, 40], totals=[10, 40, 120], prior=fitted_prior)
    assert len(a) == 3
    assert len(b) == 3
    assert np.all(np.asarray(a) > 0)
    assert np.all(np.asarray(b) > 0)


def test_posterior_alpha_exceeds_successes(fitted_prior):
    """Beta posterior alpha must be at least successes + prior.alpha."""
    a, _ = posterior_parameters(successes=[3, 8, 40], totals=[10, 40, 120], prior=fitted_prior)
    # prior alpha is positive, so posterior alpha > successes
    assert np.all(np.asarray(a) > np.array([3, 8, 40]))


def test_add_ebb_estimate_adds_columns(fitted_prior):
    """add_ebb_estimate should add ebb_fitted, ebb_low, ebb_high."""
    df = pd.DataFrame({"hits": [3, 8, 40], "at_bats": [10, 40, 120]})
    out = add_ebb_estimate(df, success_col="hits", total_col="at_bats", prior=fitted_prior)
    assert "ebb_fitted" in out.columns
    assert "ebb_low" in out.columns
    assert "ebb_high" in out.columns


def test_add_ebb_estimate_credible_interval_contains_point(fitted_prior):
    """ebb_low <= ebb_fitted <= ebb_high for every row."""
    df = pd.DataFrame({"hits": [3, 8, 40], "at_bats": [10, 40, 120]})
    out = add_ebb_estimate(df, success_col="hits", total_col="at_bats", prior=fitted_prior)
    assert np.all(out["ebb_low"] <= out["ebb_fitted"])
    assert np.all(out["ebb_fitted"] <= out["ebb_high"])


def test_add_ebb_estimate_shrinks_noisy_proportions(fitted_prior):
    """Shrunken estimate for small-N row should be closer to overall mean than raw."""
    df = pd.DataFrame({"hits": [3, 8, 40], "at_bats": [10, 40, 120]})
    out = add_ebb_estimate(df, success_col="hits", total_col="at_bats", prior=fitted_prior)
    raw = df["hits"] / df["at_bats"]
    # Shrinkage: shrunken value should be closer to overall mean than raw value
    overall = raw.mean()
    raw_distance = (raw - overall).abs()
    shrunk_distance = (out["ebb_fitted"] - overall).abs()
    # At least the smallest-N observation should be pulled toward the mean
    smallest_n_idx = df["at_bats"].idxmin()
    assert shrunk_distance.iloc[smallest_n_idx] <= raw_distance.iloc[smallest_n_idx]