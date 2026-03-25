# pyebbr

A Python port of David Robinson's [`ebbr`](https://github.com/dgrtwo/ebbr) R package for empirical Bayes binomial estimation. The goal is feature parity with the original toolkit—fit beta priors, shrink noisy proportions, run Bayesian proportion tests, and experiment with beta-mixture priors—while feeling natural to Python users (NumPy/Pandas APIs, SciPy under the hood).

## Installation

```bash
pip install pyebbr  # once published
# or, from source
pip install -e .
```

## Quick start

```python
import pandas as pd
from pyebbr import fit_beta_prior, add_ebb_estimate, ebb_prop_test

# toy data
players = pd.DataFrame({"hits": [3, 8, 40], "at_bats": [10, 40, 120]})

prior = fit_beta_prior(players["hits"], players["at_bats"])  # Beta(alpha, beta)
posterior = add_ebb_estimate(players, success_col="hits", total_col="at_bats", prior=prior)

print(posterior[["hits", "at_bats", "ebb_fitted", "ebb_low", "ebb_high"]])

# Test whether player 1's rate exceeds 0.250 after shrinkage
result = ebb_prop_test(successes=players.loc[0, "hits"], totals=players.loc[0, "at_bats"],
                       prior=prior, threshold=0.250)
print(result.prob_greater)
```

## What’s included

| Function | Description |
| --- | --- |
| `fit_beta_prior` | Maximum-likelihood beta prior fit for aggregated binomial data (with method-of-moments fallback + diagnostics). |
| `add_ebb_estimate` | DataFrame-friendly posterior updater that adds raw, shrunken, and credible interval columns. |
| `ebb_prop_test` | Empirical Bayes proportion tests (one- or two-sample). |
| `fit_beta_mixture` | EM-style fitting of multi-component beta mixtures for heavier-tailed priors. |
| `posterior_parameters` | Lightweight helper if you just want alpha/beta posteriors. |

Full API docs live in [docs/index.md](docs/index.md).

## License

MIT, matching the spirit of the original R package.
