"""Beta mixture fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln


@dataclass
class BetaMixturePrior:
    weights: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    n_components: int

    def component(self, idx: int) -> Tuple[float, float, float]:
        return (self.weights[idx], self.alphas[idx], self.betas[idx])


def _component_loglik(alpha: float, beta: float, successes: np.ndarray, totals: np.ndarray) -> np.ndarray:
    return betaln(successes + alpha, totals - successes + beta) - betaln(alpha, beta)


def _update_component(successes, totals, resp) -> Tuple[float, float]:
    def objective(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        return -np.sum(resp * _component_loglik(alpha, beta, successes, totals))

    mean = np.average(successes / totals, weights=resp)
    var = np.average(((successes / totals) - mean) ** 2, weights=resp) + 1e-6
    k = mean * (1 - mean) / var - 1
    k = max(k, 1e-3)
    x0 = np.array([mean * k, (1 - mean) * k])
    res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=((1e-6, None), (1e-6, None)))
    if not res.success:
        return x0[0], x0[1]
    return res.x[0], res.x[1]


def fit_beta_mixture(
    successes: Iterable[int],
    totals: Iterable[int],
    *,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> BetaMixturePrior:
    """Fit a beta-binomial mixture via EM."""

    if n_components < 2:
        raise ValueError("Use fit_beta_prior for single-component models")

    successes = np.asarray(list(successes), dtype=float)
    totals = np.asarray(list(totals), dtype=float)
    rates = successes / totals
    quantiles = np.quantile(rates, np.linspace(0.1, 0.9, n_components))
    weights = np.full(n_components, 1 / n_components)
    alphas = np.clip(quantiles * 10, 0.5, None)
    betas = np.clip((1 - quantiles) * 10, 0.5, None)

    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step
        log_probs = np.vstack([
            np.log(weights[j]) + _component_loglik(alphas[j], betas[j], successes, totals)
            for j in range(n_components)
        ])
        log_probs -= log_probs.max(axis=0)
        probs = np.exp(log_probs)
        resp = probs / probs.sum(axis=0, keepdims=True)

        # M-step
        weights = resp.mean(axis=1)
        for j in range(n_components):
            alphas[j], betas[j] = _update_component(successes, totals, resp[j])

        ll = np.sum(np.log((weights[:, None] * np.exp(log_probs)).sum(axis=0)))
        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return BetaMixturePrior(weights=weights, alphas=alphas, betas=betas, n_components=n_components)
