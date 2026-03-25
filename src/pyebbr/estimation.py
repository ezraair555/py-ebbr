"""Posterior estimation helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from .prior import BetaPrior, fit_beta_prior


def posterior_parameters(successes: Iterable[int], totals: Iterable[int], prior: BetaPrior):
    successes = np.asarray(list(successes), dtype=float)
    totals = np.asarray(list(totals), dtype=float)
    alpha1 = prior.alpha + successes
    beta1 = prior.beta + totals - successes
    return alpha1, beta1


def add_ebb_estimate(
    df: pd.DataFrame,
    *,
    success_col: str,
    total_col: str,
    prior: Optional[BetaPrior] = None,
    cred_level: float = 0.95,
    prefix: str = "ebb",
) -> pd.DataFrame:
    """Add empirical Bayes estimates + intervals to a DataFrame."""

    prior = prior or fit_beta_prior(df[success_col], df[total_col])
    alpha1, beta1 = posterior_parameters(df[success_col], df[total_col], prior)
    posterior_mean = alpha1 / (alpha1 + beta1)
    raw = df[success_col] / df[total_col]
    alpha = (1 - cred_level) / 2
    lower = beta_dist.ppf(alpha, alpha1, beta1)
    upper = beta_dist.ppf(1 - alpha, alpha1, beta1)

    out = df.copy()
    out[f"{prefix}_raw"] = raw
    out[f"{prefix}_alpha1"] = alpha1
    out[f"{prefix}_beta1"] = beta1
    out[f"{prefix}_fitted"] = posterior_mean
    out[f"{prefix}_low"] = lower
    out[f"{prefix}_high"] = upper
    out.attrs["ebb_prior"] = prior
    return out
