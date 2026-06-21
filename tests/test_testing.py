"""Tests for pyebbr.testing module (added 2026-06-21, Lane 4 of grade recovery)."""
import numpy as np
import pytest
from pyebbr import fit_beta_prior, ebb_prop_test, PropTestResult


@pytest.fixture
def fitted_prior():
    """Fitted prior on data with sufficient N to avoid NaN alphas."""
    return fit_beta_prior(successes=[3, 8, 40, 25, 60], totals=[10, 40, 120, 80, 200])


def test_prop_test_returns_result(fitted_prior):
    """ebb_prop_test should return a PropTestResult."""
    result = ebb_prop_test(
        successes=8, totals=40, prior=fitted_prior, threshold=0.250
    )
    assert isinstance(result, PropTestResult)


def test_prop_test_prob_greater_in_unit_interval(fitted_prior):
    """prob_greater should be a probability in [0, 1]."""
    result = ebb_prop_test(
        successes=8, totals=40, prior=fitted_prior, threshold=0.250
    )
    assert 0.0 <= result.prob_greater <= 1.0


def test_prop_test_two_sample_uses_other_kwarg(fitted_prior):
    """Two-sample comparison takes the 'other' kwarg as a (s, n) tuple."""
    result = ebb_prop_test(
        successes=8, totals=40,
        prior=fitted_prior, other=(40, 120)
    )
    assert isinstance(result, PropTestResult)
    # Two-sample returns prob_difference_gt_zero in [0, 1]
    assert 0.0 <= result.prob_difference_gt_zero <= 1.0


def test_prop_test_requires_threshold_or_other(fitted_prior):
    """ebb_prop_test should raise if neither threshold nor other is supplied."""
    with pytest.raises(ValueError, match="threshold|other|posterior"):
        ebb_prop_test(successes=8, totals=40, prior=fitted_prior)


def test_prop_test_result_has_documented_fields(fitted_prior):
    """PropTestResult should expose prob_greater / prob_lesser / prob_difference_gt_zero."""
    result = ebb_prop_test(
        successes=8, totals=40,
        prior=fitted_prior, threshold=0.250
    )
    assert hasattr(result, "prob_greater")
    assert hasattr(result, "prob_lesser")
    assert hasattr(result, "prob_difference_gt_zero")