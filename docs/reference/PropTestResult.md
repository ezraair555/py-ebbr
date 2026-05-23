# Proportion Test Result • PropTestResult

## Description

`PropTestResult` is a dataclass that stores the results of an empirical Bayes proportion test.

## Usage

```python
@dataclass
class PropTestResult:
    alpha1: float
    beta1: float
    prob_greater: Optional[float] = None
    prob_lesser: Optional[float] = None
    prob_difference_gt_zero: Optional[float] = None
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha1` | float | Posterior alpha parameter |
| `beta1` | float | Posterior beta parameter |
| `prob_greater` | Optional[float] | P(rate > threshold) or P(rate1 > rate2) |
| `prob_lesser` | Optional[float] | P(rate < threshold) |
| `prob_difference_gt_zero` | Optional[float] | Alias for prob_greater in comparison tests |

## Details

This dataclass is returned by [`ebb_prop_test()`](ebb_prop_test.html) and contains both the posterior parameters and the hypothesis test results.

**Test Types:**

1. **Threshold Test** (when `threshold` is provided):
   - `prob_greater`: Probability that true rate > threshold
   - `prob_lesser`: Probability that true rate < threshold
   - Note: prob_greater + prob_lesser = 1.0

2. **Comparison Test** (when `other` is provided):
   - `prob_difference_gt_zero`: Probability that rate1 > rate2
   - Computed via Monte Carlo sampling (20,000 samples)

## Examples

### Threshold Test Result

```python
from pyebbr import fit_beta_prior, ebb_prop_test

# Fit prior
df = pd.DataFrame({
    'hits': [45, 30, 15],
    'at_bats': [150, 100, 50]
})
prior = fit_beta_prior(df['hits'], df['at_bats'])

# Test against threshold
result = ebb_prop_test(
    successes=45,
    totals=150,
    prior=prior,
    threshold=0.300
)

print(f"Posterior alpha: {result.alpha1:.2f}")
print(f"Posterior beta: {result.beta1:.2f}")
print(f"P(batting avg > .300): {result.prob_greater:.3f}")
print(f"P(batting avg < .300): {result.prob_lesser:.3f}")
```

### Comparison Test Result

```python
# Compare two players
result = ebb_prop_test(
    successes=45,
    totals=150,
    prior=prior,
    other=(30, 100)  # Compare to Johnson
)

print(f"P(Smith > Johnson): {result.prob_difference_gt_zero:.3f}")
```

### Interpret Test Results

```python
result = ebb_prop_test(
    successes=45,
    totals=150,
    prior=prior,
    threshold=0.300
)

# Interpret probability
if result.prob_greater > 0.95:
    conclusion = "Very strong evidence"
elif result.prob_greater > 0.75:
    conclusion = "Moderate evidence"
elif result.prob_greater > 0.50:
    conclusion = "Weak evidence"
else:
    conclusion = "No evidence"

print(f"{conclusion} that true rate > 0.300")
```

### Extract and Use Results

```python
# Test multiple players
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
        'prob_gt_300': result.prob_greater,
        'posterior_mean': result.alpha1 / (result.alpha1 + result.beta1)
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('prob_gt_300', ascending=False))
```

## See Also

- [`ebb_prop_test()`](ebb_prop_test.html) - Perform proportion test
- [`BetaPrior`](BetaPrior.html) - Prior container
- [`add_ebb_estimate()`](add_ebb_estimate.html) - Add EB estimates
