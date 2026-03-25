"""Empirical Bayes proportion tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.stats import beta as beta_dist

from .prior import BetaPrior
from .estimation import posterior_parameters


@dataclass
class PropTestResult:
    alpha1: float
    beta1: float
    prob_greater: Optional[float] = None
    prob_lesser: Optional[float] = None
    prob_difference_gt_zero: Optional[float] = None


def ebb_prop_test(
    *,
    successes: int,
    totals: int,
    prior: BetaPrior,
    threshold: Optional[float] = None,
    other: Optional[Tuple[int, int]] = None,
) -> PropTestResult:
    """Posterior probability that a true rate exceeds a threshold or another posterior."""

    alpha1, beta1 = posterior_parameters([successes], [totals], prior)
    alpha1, beta1 = alpha1[0], beta1[0]

    if other is not None:
        s2, n2 = other
        alpha2, beta2 = posterior_parameters([s2], [n2], prior)
        alpha2, beta2 = alpha2[0], beta2[0]
        # Monte Carlo comparison of two beta posteriors
        samples1 = beta_dist.rvs(alpha1, beta1, size=20000)
        samples2 = beta_dist.rvs(alpha2, beta2, size=20000)
        diff_prob = float((samples1 > samples2).mean())
        return PropTestResult(alpha1, beta1, prob_difference_gt_zero=diff_prob)

    if threshold is None:
        raise ValueError("Provide either a threshold or 'other' posterior to compare against.")

    prob_greater = 1 - beta_dist.cdf(threshold, alpha1, beta1)
    prob_lesser = beta_dist.cdf(threshold, alpha1, beta1)
    return PropTestResult(alpha1, beta1, prob_greater=float(prob_greater), prob_lesser=float(prob_lesser))
