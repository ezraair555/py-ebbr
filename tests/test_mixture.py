"""Tests for pyebbr.mixture module (added 2026-06-21, Lane 4 of grade recovery)."""
import numpy as np
import pytest
from pyebbr.mixture import BetaMixturePrior, fit_beta_mixture


def test_fit_beta_mixture_rejects_single_component():
    """fit_beta_mixture requires >= 2 components (use fit_beta_prior for 1)."""
    s = np.array([3, 8, 40, 12, 25])
    t = np.array([10, 40, 120, 50, 80])
    with pytest.raises(ValueError, match="single-component|fit_beta_prior"):
        fit_beta_mixture(s, t, n_components=1)


def test_fit_beta_mixture_two_components_shapes():
    """A 2-component mixture should have 2 weights + 2 (alpha, beta) pairs."""
    s = np.array([3, 8, 40, 12, 25, 5, 60])
    t = np.array([10, 40, 120, 50, 80, 30, 200])
    prior = fit_beta_mixture(s, t, n_components=2)
    assert len(prior.weights) == 2
    assert len(prior.alphas) == 2
    assert len(prior.betas) == 2


def test_beta_mixture_prior_weights_sum_to_one():
    """Mixture weights should be a proper probability simplex (sum to 1)."""
    s = np.array([3, 8, 40, 12, 25, 5, 60])
    t = np.array([10, 40, 120, 50, 80, 30, 200])
    prior = fit_beta_mixture(s, t, n_components=3)
    assert abs(sum(prior.weights) - 1.0) < 1e-6


def test_beta_mixture_prior_components_positive():
    """All alpha and beta parameters should be strictly positive."""
    s = np.array([3, 8, 40, 12, 25])
    t = np.array([10, 40, 120, 50, 80])
    prior = fit_beta_mixture(s, t, n_components=2)
    assert all(a > 0 for a in prior.alphas)
    assert all(b > 0 for b in prior.betas)