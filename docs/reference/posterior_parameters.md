# Posterior Parameters • posterior_parameters

## Description

`posterior_parameters()` computes the posterior alpha and beta parameters for binomial observations given a beta prior.

## Usage

```python
posterior_parameters(
    successes: Iterable[int],
    totals: Iterable[int],
    prior: BetaPrior,
) -> Tuple[np.ndarray, np.ndarray]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `successes` | Iterable of success counts |
| `totals` | Iterable of total trial counts |
| `prior` | Fitted `BetaPrior` object |

## Details

In Bayesian inference with a beta prior and binomial likelihood, the posterior distribution is also a beta distribution with updated parameters:

- **Posterior alpha**: α_posterior = α_prior + successes
- **Posterior beta**: β_posterior = β_prior + (totals - successes)

This is a conjugate prior relationship, meaning the posterior has the same functional form as the prior.

The posterior mean (empirical Bayes estimate) is:
- **Mean**: α_posterior / (α_posterior + β_posterior)

## Returns

A tuple of two numpy arrays:
- `alpha1`: Posterior alpha parameters
- `beta1`: Posterior beta parameters

## Examples

### Compute Posterior Parameters

```python
import pandas as pd
from pyebbr import fit_beta_prior, posterior_parameters

# Baseball data
df = pd.DataFrame({
    'player': ['Smith', 'Johnson', 'Williams'],
    'hits': [45, 30, 15],
    'at_bats': [150, 100, 50]
})

# Fit prior
prior = fit_beta_prior(df['hits'], df['at_bats'])

# Compute posterior parameters
alpha1, beta1 = posterior_parameters(df['hits'], df['at_bats'], prior)

print(f"Posterior alpha: {alpha1}")
print(f"Posterior beta: {beta1}")
```

### Calculate Posterior Means

```python
# Compute posterior parameters
alpha1, beta1 = posterior_parameters(df['hits'], df['at_bats'], prior)

# Calculate posterior means (EB estimates)
posterior_means = alpha1 / (alpha1 + beta1)

print("Posterior means (EB estimates):")
for player, mean in zip(df['player'], posterior_means):
    print(f"  {player}: {mean:.3f}")
```

### Compare Raw vs. Posterior Estimates

```python
alpha1, beta1 = posterior_parameters(df['hits'], df['at_bats'], prior)

raw_rates = df['hits'] / df['at_bats']
posterior_means = alpha1 / (alpha1 + beta1)

comparison = pd.DataFrame({
    'player': df['player'],
    'raw_rate': raw_rates,
    'eb_estimate': posterior_means,
    'shrinkage': posterior_means - raw_rates
})

print(comparison)
# Notice: small samples shrink more toward the prior mean
```

### Use with Single Observation

```python
# Test a single new observation
new_hits = 12
new_abs = 40

alpha1, beta1 = posterior_parameters([new_hits], [new_abs], prior)
posterior_mean = alpha1[0] / (alpha1[0] + beta1[0])

print(f"New observation EB estimate: {posterior_mean:.3f}")
```

## See Also

- [`fit_beta_prior()`](fit_beta_prior.html) - Fit beta prior
- [`add_ebb_estimate()`](add_ebb_estimate.html) - Add EB estimates (uses this internally)
- [`BetaPrior`](BetaPrior.html) - Prior container
