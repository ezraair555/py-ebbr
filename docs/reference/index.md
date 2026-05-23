# Package Index • py-ebbr

---

## Fitting Priors

These functions fit beta priors to binomial data using empirical Bayes methods.

| Function | Description |
|----------|-------------|
| [`fit_beta_prior()`](fit_beta_prior.html) | Fit a Beta prior from binomial observations |
| [`fit_beta_mixture()`](fit_beta_mixture.html) | Fit a beta-binomial mixture via EM algorithm |
| [`BetaPrior`](BetaPrior.html) | Container for a fitted beta prior |
| [`BetaMixturePrior`](BetaMixturePrior.html) | Container for a fitted beta mixture prior |

---

## Estimation

Functions for computing posterior estimates and adding them to data frames.

| Function | Description |
|----------|-------------|
| [`add_ebb_estimate()`](add_ebb_estimate.html) | Add empirical Bayes estimates + intervals to a DataFrame |
| [`posterior_parameters()`](posterior_parameters.html) | Compute posterior alpha and beta parameters |

---

## Hypothesis Testing

Functions for empirical Bayes hypothesis testing and comparisons.

| Function | Description |
|----------|-------------|
| [`ebb_prop_test()`](ebb_prop_test.html) | Test whether a proportion exceeds a threshold or another proportion |
| [`PropTestResult`](PropTestResult.html) | Result of an empirical Bayes proportion test |

---

## Installation

```bash
pip install py-ebbr
```

## Quick Start

```python
import pandas as pd
from pyebbr import fit_beta_prior, add_ebb_estimate

# Baseball batting data
df = pd.DataFrame({
    'player': ['A', 'B', 'C', 'D'],
    'hits': [45, 30, 15, 60],
    'at_bats': [150, 100, 50, 200]
})

# Fit beta prior
prior = fit_beta_prior(df['hits'], df['at_bats'])

# Add empirical Bayes estimates
df_with_eb = add_ebb_estimate(df, success_col='hits', total_col='at_bats', prior=prior)

print(df_with_eb)
```

---

## See Also

- [GitHub Repository](https://github.com/ezraair555/py-ebbr)
- [Quick Start Notebook](../notebooks/quickstart.html)
- [Original R Package](https://cran.r-project.org/package=ebbr)

---

*py-ebbr is a Python port of David Robinson's R `ebbr` package for empirical Bayes estimation of binomial data.*
