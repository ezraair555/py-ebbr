# Fit Beta Prior • fit_beta_prior

## Description

`fit_beta_prior()` fits a Beta prior distribution to binomial observations using maximum likelihood estimation (MLE) or method of moments.

## Usage

```python
fit_beta_prior(
    successes: Iterable[int],
    totals: Iterable[int],
    *,
    method: str = "mle",
    initial: Optional[Tuple[float, float]] = None,
) -> BetaPrior
```

## Arguments

| Argument | Description |
|----------|-------------|
| `successes` | Iterable of success counts (e.g., hits, conversions) |
| `totals` | Iterable of total trial counts (e.g., at-bats, impressions) |
| `method` | Estimation method: `"mle"` (default) or `"moments"` |
| `initial` | Optional initial values for MLE optimization |

## Details

The beta-binomial model assumes that success rates follow a beta distribution with parameters α (alpha) and β (beta). This function estimates these parameters from observed binomial data.

**Estimation Methods:**

- **MLE** (default): Maximizes the beta-binomial likelihood using L-BFGS-B optimization. More accurate but slower. Falls back to method of moments if optimization fails.

- **Moments**: Uses method of moments estimation based on mean and variance of observed rates. Faster but less accurate for small samples.

**Constraints:**
- All `totals` must be positive
- All `successes` must be in range [0, totals]
- `successes` and `totals` must have the same length

## Returns

A `BetaPrior` dataclass with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha` | float | Estimated alpha parameter |
| `beta` | float | Estimated beta parameter |
| `n_obs` | int | Number of observations used |
| `method` | str | Estimation method used ("mle" or "moments") |
| `mean` | float | Prior mean: alpha / (alpha + beta) |
| `variance` | float | Prior variance |

## Examples

### Fit Prior to Baseball Data

```python
import pandas as pd
from pyebbr import fit_beta_prior

# Baseball batting data
df = pd.DataFrame({
    'player': ['Smith', 'Johnson', 'Williams', 'Brown'],
    'hits': [45, 30, 15, 60],
    'at_bats': [150, 100, 50, 200]
})

# Fit beta prior using MLE (default)
prior = fit_beta_prior(df['hits'], df['at_bats'])

print(f"Alpha: {prior.alpha:.3f}")
print(f"Beta: {prior.beta:.3f}")
print(f"Prior mean: {prior.mean:.3f}")
print(f"Prior variance: {prior.variance:.6f}")
```

### Compare Estimation Methods

```python
# MLE estimation (default)
prior_mle = fit_beta_prior(df['hits'], df['at_bats'], method="mle")

# Method of moments
prior_mom = fit_beta_prior(df['hits'], df['at_bats'], method="moments")

print(f"MLE:      alpha={prior_mle.alpha:.3f}, beta={prior_mle.beta:.3f}")
print(f"Moments:  alpha={prior_mom.alpha:.3f}, beta={prior_mom.beta:.3f}")
```

### Provide Initial Values

```python
# Provide custom initial values for MLE
initial = (50.0, 100.0)  # (alpha, beta)
prior = fit_beta_prior(df['hits'], df['at_bats'], initial=initial)
```

### Handle Edge Cases

```python
# Small sample sizes
small_df = pd.DataFrame({
    'hits': [1, 0, 2],
    'at_bats': [5, 3, 10]
})

prior = fit_beta_prior(small_df['hits'], small_df['at_bats'])
print(f"Small sample prior: mean={prior.mean:.3f}")
```

## See Also

- [`BetaPrior`](BetaPrior.html) - Container for fitted prior
- [`add_ebb_estimate()`](add_ebb_estimate.html) - Add EB estimates to data
- [`fit_beta_mixture()`](fit_beta_mixture.html) - Fit beta mixture prior
- [`posterior_parameters()`](posterior_parameters.html) - Compute posterior parameters

## References

Ported from David Robinson's R `ebbr` package: https://cran.r-project.org/package=ebbr
