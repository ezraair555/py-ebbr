# Beta Prior Container • BetaPrior

## Description

`BetaPrior` is a dataclass that stores the parameters of a fitted beta distribution for empirical Bayes estimation.

## Usage

```python
@dataclass
class BetaPrior:
    alpha: float
    beta: float
    n_obs: int
    method: str = "mle"
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha` | float | Alpha (shape) parameter of the beta distribution |
| `beta` | float | Beta (shape) parameter of the beta distribution |
| `n_obs` | int | Number of observations used to fit the prior |
| `method` | str | Estimation method: "mle" or "moments" |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `mean` | float | Prior mean: α / (α + β) |
| `variance` | float | Prior variance: (αβ) / ((α+β)²(α+β+1)) |

## Details

The beta distribution is parameterized by two shape parameters, α (alpha) and β (beta), which determine the distribution's shape:

- **Mean**: α / (α + β) - the expected value
- **Variance**: decreases as α + β increases (more data = more certainty)

**Interpretation:**
- α + β represents the "sample size" or strength of the prior
- Larger α + β = stronger prior (more shrinkage toward the mean)
- Smaller α + β = weaker prior (less shrinkage)

## Examples

### Create BetaPrior Directly

```python
from pyebbr import BetaPrior

# Create a prior with alpha=50, beta=100
prior = BetaPrior(alpha=50, beta=100, n_obs=100, method="mle")

print(f"Prior mean: {prior.mean:.3f}")
print(f"Prior variance: {prior.variance:.6f}")
```

### Access Prior Properties

```python
from pyebbr import fit_beta_prior

# Fit prior to data
df = pd.DataFrame({
    'hits': [45, 30, 15, 60],
    'at_bats': [150, 100, 50, 200]
})

prior = fit_beta_prior(df['hits'], df['at_bats'])

print(f"Alpha: {prior.alpha}")
print(f"Beta: {prior.beta}")
print(f"Number of observations: {prior.n_obs}")
print(f"Estimation method: {prior.method}")
print(f"Mean: {prior.mean:.3f}")
print(f"Variance: {prior.variance:.6f}")
```

### Compare Priors

```python
# Strong prior (large alpha + beta)
strong_prior = BetaPrior(alpha=500, beta=1000, n_obs=1500)

# Weak prior (small alpha + beta)
weak_prior = BetaPrior(alpha=5, beta=10, n_obs=15)

print(f"Strong prior mean: {strong_prior.mean:.3f}, variance: {strong_prior.variance:.6f}")
print(f"Weak prior mean: {weak_prior.mean:.3f}, variance: {weak_prior.variance:.6f}")

# Notice: same mean, but strong prior has much smaller variance
```

### Use in Empirical Bayes Estimation

```python
from pyebbr import BetaPrior, add_ebb_estimate

# Define prior based on historical data
prior = BetaPrior(alpha=75, beta=175, n_obs=250, method="mle")

# Apply to new data
df = pd.DataFrame({
    'player': ['New1', 'New2'],
    'hits': [12, 8],
    'at_bats': [40, 30]
})

df_with_eb = add_ebb_estimate(df, success_col='hits', total_col='at_bats', prior=prior)
print(df_with_eb)
```

## See Also

- [`fit_beta_prior()`](fit_beta_prior.html) - Fit beta prior from data
- [`BetaMixturePrior`](BetaMixturePrior.html) - Mixture of beta priors
- [`add_ebb_estimate()`](add_ebb_estimate.html) - Use prior for estimation
