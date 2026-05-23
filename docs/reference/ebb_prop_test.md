# Empirical Bayes Proportion Test • ebb_prop_test

## Description

`ebb_prop_test()` performs empirical Bayes hypothesis testing for binomial proportions. It computes the posterior probability that a true rate exceeds a threshold or is greater than another proportion.

## Usage

```python
ebb_prop_test(
    *,
    successes: int,
    totals: int,
    prior: BetaPrior,
    threshold: Optional[float] = None,
    other: Optional[Tuple[int, int]] = None,
) -> PropTestResult
```

## Arguments

| Argument | Description |
|----------|-------------|
| `successes` | Number of successes for the observation being tested |
| `totals` | Total number of trials for the observation |
| `prior` | Fitted `BetaPrior` object from `fit_beta_prior()` |
| `threshold` | Threshold value to test against (e.g., 0.300 for batting average) |
| `other` | Alternative: tuple of (successes, totals) to compare against |

## Details

This function performs Bayesian hypothesis testing using the posterior distribution derived from empirical Bayes estimation.

**Two Test Types:**

1. **Threshold Test**: Tests whether the true rate exceeds a specified threshold
   - Returns probability that rate > threshold
   - Returns probability that rate < threshold

2. **Comparison Test**: Tests whether one rate is greater than another
   - Uses Monte Carlo sampling (20,000 samples) to compare two beta posteriors
   - Returns probability that rate1 > rate2

**Note:** You must provide either `threshold` or `other`, but not both.

## Returns

A `PropTestResult` dataclass with the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha1` | float | Posterior alpha parameter |
| `beta1` | float | Posterior beta parameter |
| `prob_greater` | Optional[float] | P(rate > threshold) or P(rate1 > rate2) |
| `prob_lesser` | Optional[float] | P(rate < threshold) |
| `prob_difference_gt_zero` | Optional[float] | Alias for prob_greater in comparison tests |

## Examples

### Test Against Threshold

```python
import pandas as pd
from pyebbr import fit_beta_prior, ebb_prop_test

# Baseball batting data
df = pd.DataFrame({
    'player': ['Smith', 'Johnson', 'Williams'],
    'hits': [45, 30, 15],
    'at_bats': [150, 100, 50]
})

# Fit prior
prior = fit_beta_prior(df['hits'], df['at_bats'])

# Test if Smith's true batting average > 0.300
result = ebb_prop_test(
    successes=45,
    totals=150,
    prior=prior,
    threshold=0.300
)

print(f"P(batting avg > .300) = {result.prob_greater:.3f}")
print(f"P(batting avg < .300) = {result.prob_lesser:.3f}")
```

### Compare Two Players

```python
# Compare Smith vs Johnson
result = ebb_prop_test(
    successes=45,  # Smith's hits
    totals=150,    # Smith's at-bats
    prior=prior,
    other=(30, 100)  # Johnson's (hits, at-bats)
)

print(f"P(Smith > Johnson) = {result.prob_difference_gt_zero:.3f}")
```

### Find Best Performers

```python
# Test all players against threshold
results = []
for _, row in df.iterrows():
    result = ebb_prop_test(
        successes=row['hits'],
        totals=row['at_bats'],
        prior=prior,
        threshold=0.300
    )
    results.append({
        'player': row['player'],
        'prob_gt_300': result.prob_greater
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('prob_gt_300', ascending=False))
```

### Interpret Results

```python
result = ebb_prop_test(
    successes=45,
    totals=150,
    prior=prior,
    threshold=0.300
)

# Strong evidence if probability > 0.95
if result.prob_greater > 0.95:
    print("Strong evidence that true rate > 0.300")
elif result.prob_greater > 0.75:
    print("Moderate evidence that true rate > 0.300")
else:
    print("Insufficient evidence")
```

## See Also

- [`fit_beta_prior()`](fit_beta_prior.html) - Fit beta prior to data
- [`add_ebb_estimate()`](add_ebb_estimate.html) - Add EB estimates to data
- [`PropTestResult`](PropTestResult.html) - Test result container

## References

Ported from David Robinson's R `ebbr` package: https://cran.r-project.org/package=ebbr
