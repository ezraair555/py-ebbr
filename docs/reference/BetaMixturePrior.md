# Beta Mixture Prior Container • BetaMixturePrior

## Description

`BetaMixturePrior` is a dataclass that stores the parameters of a fitted beta mixture model with multiple components.

## Usage

```python
@dataclass
class BetaMixturePrior:
    weights: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    n_components: int
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | np.ndarray | Mixture weights for each component (sum to 1) |
| `alphas` | np.ndarray | Alpha parameters for each component |
| `betas` | np.ndarray | Beta parameters for each component |
| `n_components` | int | Number of mixture components |

## Methods

| Method | Description |
|--------|-------------|
| `component(idx)` | Returns (weight, alpha, beta) tuple for component `idx` |

## Details

The beta mixture model represents the prior as a weighted combination of K beta distributions:

```
p(θ) = Σ w_k × Beta(θ | α_k, β_k)
```

where:
- `w_k` is the weight of component k (Σ w_k = 1)
- Each component has its own alpha and beta parameters

This allows modeling of multimodal or heterogeneous populations where a single beta distribution would be inadequate.

## Examples

### Create BetaMixturePrior

```python
import numpy as np
from pyebbr import BetaMixturePrior

# Create a 2-component mixture
mixture = BetaMixturePrior(
    weights=np.array([0.7, 0.3]),
    alphas=np.array([50, 20]),
    betas=np.array([100, 80]),
    n_components=2
)

print(f"Component 0: weight={mixture.weights[0]}, "
      f"alpha={mixture.alphas[0]}, beta={mixture.betas[0]}")
print(f"Component 1: weight={mixture.weights[1]}, "
      f"alpha={mixture.alphas[1]}, beta={mixture.betas[1]}")
```

### Access Component Parameters

```python
from pyebbr import fit_beta_mixture

# Fit mixture to data
df = pd.DataFrame({
    'hits': [45, 30, 15, 5, 8, 3],
    'at_bats': [150, 100, 50, 50, 40, 30]
})

mixture = fit_beta_mixture(df['hits'], df['at_bats'], n_components=2)

# Get specific component
weight, alpha, beta = mixture.component(0)
print(f"Component 0: weight={weight}, alpha={alpha}, beta={beta}")
```

### Calculate Component Means

```python
mixture = fit_beta_mixture(df['hits'], df['at_bats'], n_components=2)

# Calculate mean for each component
for i in range(mixture.n_components):
    mean = mixture.alphas[i] / (mixture.alphas[i] + mixture.betas[i])
    print(f"Component {i}: weight={mixture.weights[i]:.2f}, mean={mean:.3f}")
```

### Identify Clusters

```python
mixture = fit_beta_mixture(df['hits'], df['at_bats'], n_components=2)

# Find high-performing component
means = [
    mixture.alphas[i] / (mixture.alphas[i] + mixture.betas[i])
    for i in range(mixture.n_components)
]
high_comp = np.argmax(means)

print(f"High-performing component: {high_comp}")
print(f"  Weight: {mixture.weights[high_comp]:.2f}")
print(f"  Mean: {means[high_comp]:.3f}")
```

### Compare Single vs. Mixture Prior

```python
from pyebbr import fit_beta_prior

# Fit single prior
single_prior = fit_beta_prior(df['hits'], df['at_bats'])

# Fit mixture prior
mixture_prior = fit_beta_mixture(df['hits'], df['at_bats'], n_components=2)

print("Single prior:")
print(f"  Mean: {single_prior.mean:.3f}")

print("\nMixture prior:")
for i in range(mixture_prior.n_components):
    mean = mixture_prior.alphas[i] / (mixture_prior.alphas[i] + mixture_prior.betas[i])
    print(f"  Component {i}: weight={mixture_prior.weights[i]:.2f}, mean={mean:.3f}")
```

## See Also

- [`fit_beta_mixture()`](fit_beta_mixture.html) - Fit beta mixture prior
- [`BetaPrior`](BetaPrior.html) - Single beta prior container
- [`fit_beta_prior()`](fit_beta_prior.html) - Fit single beta prior
