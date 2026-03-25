# pyebbr Documentation

pyebbr mirrors the conceptual flow of the original `ebbr` R package:

1. **Fit a prior** – learn a Beta(α, β) (or mixture) that represents the background rate of success.
2. **Update each observation** – combine the prior with each success/total pair to get a posterior.
3. **Summarize / test** – shrink estimates, compute credible intervals, or run posterior probability tests.

## API Overview

### `fit_beta_prior(successes, totals, *, method="mle", initial=None)`
- Uses maximum likelihood to fit a beta prior to parallel arrays of successes and totals.
- Falls back to method-of-moments when MLE fails.
- Returns a `BetaPrior(alpha, beta, n_observations, method)` dataclass.

### `posterior_parameters(successes, totals, prior)`
- Vectorized helper returning `(alpha1, beta1)` arrays (posterior parameters) for each observation.

### `add_ebb_estimate(df, success_col, total_col, prior=None, cred_level=0.95, prefix="ebb")`
- Adds `.raw`, `.alpha1`, `.beta1`, `.fitted`, `.low`, `.high` columns to a copy of `df`.
- If `prior=None`, it first fits one via `fit_beta_prior`.

### `ebb_prop_test(successes, totals, prior, *, threshold=None, other=None)`
- One-sample: set `threshold` to test `P(p > threshold | data)`.
- Two-sample: pass `other=(success2, total2)` to compare two posteriors.

### `fit_beta_mixture(successes, totals, n_components=2, max_iter=200, tol=1e-6)`
- EM algorithm over beta-binomial mixtures.
- Returns weights + component priors for downstream use.

## Examples

See [`examples/notebooks`](../examples) for richer demos (fitting baseball batting averages, A/B email tests, etc.).

## Relationship to `ebbr`

- Covers the same core helpers: `ebb_fit_prior`, `add_ebb_estimate`, `add_ebb_prop_test`, `ebb_fit_mixture`.
- Naming follows Python conventions while keeping arguments close to the R originals for familiarity.
- Additional conveniences (NumPy arrays, Pandas DataFrames, dataclasses) are exposed so other skills/scripts can import them directly.
