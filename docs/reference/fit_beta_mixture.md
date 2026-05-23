# Fit Beta Mixture Prior • fit_beta_mixture

## Description

`fit_beta_mixture()` fits a beta-binomial mixture model using the EM (Expectation-Maximization) algorithm. This is useful when the data comes from multiple underlying populations with different success rates.

## Usage

```python
fit_beta_mixture(
    successes: Iterable[int],
    totals: Iterable[int],
    *,
    n_components: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> BetaMixturePrior
```

## Arguments

| Argument | Description |
|----------|-------------|
| `successes` | Iterable of success counts |
| `totals` | Iterable of total trial counts |
| `n_components` | Number of mixture components (default: 2) |
| `max_iter` | Maximum number of EM iterations (default: 200) |
| `tol` | Convergence tolerance (default: 1e-6) |

## Details

The beta mixture model assumes that the data comes from a mixture of K beta distributions, each representing a different subpopulation. This is more flexible than a single beta prior when the data is multimodal.

**EM Algorithm:**
1. **E-step**: Compute responsibility of each component for each observation
2. **M-step**: Update component parameters (weights, alpha, beta) based on responsibilities
3. Repeat until convergence (change in log-likelihood < tolerance)

**Use Cases:**
- Identifying high/low performers in a population
- Modeling heterogeneous groups (e.g., different player positions)
- Clustering binomial data without explicit labels

## Returns

A `BetaMixturePrior` dataclass with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | np.ndarray | Mixture weights (sum to 1) |
| `alphas` | np.ndarray | Alpha parameters for each component |
| `betas` | np.ndarray | Beta parameters for each component |
| `n_components` | int | Number of components |

## Examples

### Fit Two-Component Mixture

```python
import pandas as pd
from pyebbr import fit_beta_mixture

# Baseball data with pitchers and hitters mixed
df = pd.DataFrame({
    'player': ['A', 'B', 'C', 'D', 'E', 'F'],
    'hits': [45, 30, 15, 5, 8, 3],
    'at_bats': [150, 100, 50, 50, 40, 30]
})

# Fit 2-component mixture
mixture = fit_beta_mixture(df['hits'], df['at_bats'], n_components=2)

print(f"Component 1: weight={mixture.weights[0]:.3f}, "
      f"alpha={mixture.alphas[0]:.2f}, beta={mixture.betas[0]:.2f}")
print(f"Component 2: weight={mixture.weights[1]:.3f}, "
      f"alpha={mixture.alphas[1]:.2f}, beta={mixture.betas[1]:.2f}")
```

### Identify High/Low Performers

```python
# Fit mixture
mixture = fit_beta_mixture(df['hits'], df['at_bats'], n_components=2)

# Calculate component means
means = [mixture.alphas[i] / (mixture.alphas[i] + mixture.betas[i])
         for i in range(mixture.n_components)]

print(f"Component means: {means}")

# Identify which component is "high performer"
high_comp = 0 if means[0] > means[1] else 1
print(f"High performer component: {high_comp} (mean={means[high_comp]:.3f})")
```

### Three-Component Mixture

```python
# Fit 3-component mixture for more granular clustering
mixture = fit_beta_mixture(
    df['hits'],
    df['at_bats'],
    n_components=3,
    max_iter=500
)

for i in range(3):
    mean = mixture.alphas[i] / (mixture.alphas[i] + mixture.betas[i])
    print(f"Component {i}: weight={mixture.weights[i]:.2f}, mean={mean:.3f}")
```

### Extract Component Parameters

```python
mixture = fit_beta_mixture(df['hits'], df['at_bats'])

# Access individual component
weight, alpha, beta = mixture.component(0)
print(f"Component 0: weight={weight}, alpha={alpha}, beta={beta}")
```

## See Also

- [`fit_beta_prior()`](fit_beta_prior.html) - Fit single beta prior
- [`BetaMixturePrior`](BetaMixturePrior.html) - Container for mixture prior
- [`add_ebb_estimate()`](add_ebb_estimate.html) - Add EB estimates (works with single prior)

## References

Ported from David Robinson's R `ebbr` package: https://cran.r-project.org/package=ebbr
