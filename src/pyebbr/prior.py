"""Prior-fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln


@dataclass
class BetaPrior:
    """Simple container for a fitted beta prior."""

    alpha: float
    beta: float
    n_obs: int
    method: str = "mle"

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))


def _beta_binomial_loglik(params: Sequence[float], successes: np.ndarray, totals: np.ndarray) -> float:
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    k = successes
    n = totals
    # log C(n,k) + betaln(k+alpha, n-k+beta) - betaln(alpha, beta)
    log_coeff = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    ll = log_coeff + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)
    return -np.sum(ll)


def _method_of_moments(successes: np.ndarray, totals: np.ndarray) -> Tuple[float, float]:
    rates = successes / totals
    mean = np.clip(rates.mean(), 1e-6, 1 - 1e-6)
    var = np.var(rates, ddof=1)
    var = max(var, 1e-6)
    common = mean * (1 - mean) / var - 1
    common = max(common, 1e-3)
    alpha = mean * common
    beta = (1 - mean) * common
    return alpha, beta


def fit_beta_prior(
    successes: Iterable[int],
    totals: Iterable[int],
    *,
    method: str = "mle",
    initial: Optional[Tuple[float, float]] = None,
) -> BetaPrior:
    """Fit a Beta prior from binomial observations."""

    successes = np.asarray(list(successes), dtype=float)
    totals = np.asarray(list(totals), dtype=float)
    if successes.shape != totals.shape:
        raise ValueError("successes and totals must align")
    if np.any(totals <= 0):
        raise ValueError("totals must be positive")
    if np.any(successes < 0) or np.any(successes > totals):
        raise ValueError("successes must be in [0, totals]")

    if method not in {"mle", "moments"}:
        raise ValueError("method must be 'mle' or 'moments'")

    if method == "moments":
        alpha, beta = _method_of_moments(successes, totals)
        return BetaPrior(alpha, beta, n_obs=len(successes), method="moments")

    x0 = initial or _method_of_moments(successes, totals)
    result = minimize(
        _beta_binomial_loglik,
        x0=np.asarray(x0),
        args=(successes, totals),
        bounds=((1e-6, None), (1e-6, None)),
        method="L-BFGS-B",
    )
    if not result.success:
        alpha, beta = _method_of_moments(successes, totals)
        method_used = "moments"
    else:
        alpha, beta = result.x
        method_used = "mle"

    return BetaPrior(float(alpha), float(beta), n_obs=len(successes), method=method_used)
