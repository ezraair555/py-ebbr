# Add Empirical Bayes Estimates • add_ebb_estimate

## Description

`add_ebb_estimate()` adds empirical Bayes estimates and credible intervals to a DataFrame containing binomial data (successes and totals).

## Usage

```python
add_ebb_estimate(
    df: pd.DataFrame,
    *,
    success_col: str,
    total_col: str,
    prior: Optional[BetaPrior] = None,
    cred_level: float = 0.95,
    prefix: str = "ebb",
) -> pd.DataFrame
```

## Arguments

| Argument | Description |
|----------|-------------|
| `df` | Pandas DataFrame with binomial data |
| `success_col` | Column name for success counts |
| `total_col` | Column name for total trial counts |
| `prior` | Optional `BetaPrior` object (fitted automatically if not provided) |
| `cred_level` | Credible interval level (default: 0.95 for 95% CI) |
| `prefix` | Prefix for new column names (default: "ebb") |

## Details

Empirical Bayes estimation "shrinks" individual estimates toward the overall mean, providing more stable estimates especially for observations with small sample sizes.

The function adds the following columns to the DataFrame:

| Column | Description |
|--------|-------------|
| `{prefix}_raw` | Raw success rate (successes / totals) |
| `{prefix}_alpha1` | Posterior alpha parameter |
| `{prefix}_beta1` | Posterior beta parameter |
| `{prefix}_fitted` | Empirical Bayes estimate (posterior mean) |
| `{prefix}_low` | Lower bound of credible interval |
| `{prefix}_high` | Upper bound of credible interval |

The prior is stored in `df.attrs["ebb_prior"]` for later reference.

## Returns

A copy of the input DataFrame with added columns for empirical Bayes estimates.

## Examples

### Add EB Estimates to Baseball Data

```python
import pandas as pd
from pyebbr import add_ebb_estimate

# Baseball batting data
df = pd.DataFrame({
    'player': ['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'],
    'hits': [45, 30, 15, 60, 5],
    'at_bats': [150, 100, 50, 200, 20]
})

# Add empirical Bayes estimates (prior fitted automatically)
df_with_eb = add_ebb_estimate(df, success_col='hits', total_col='at_bats')

print(df_with_eb)
```

### Compare Raw vs. EB Estimates

```python
df_with_eb = add_ebb_estimate(df, success_col='hits', total_col='at_bats')

# Display comparison
print(df_with_eb[['player', 'ebb_raw', 'ebb_fitted']])

# Notice how small samples are shrunk toward the mean:
# Davis (5/20 = .250) might have EB estimate closer to .300
```

### Use Custom Prior

```python
from pyebbr import fit_beta_prior, add_ebb_estimate

# Fit prior separately
prior = fit_beta_prior(df['hits'], df['at_bats'])

# Add estimates with custom prior
df_with_eb = add_ebb_estimate(
    df,
    success_col='hits',
    total_col='at_bats',
    prior=prior
)
```

### Adjust Credible Interval Level

```python
# 90% credible intervals instead of 95%
df_with_eb = add_ebb_estimate(
    df,
    success_col='hits',
    total_col='at_bats',
    cred_level=0.90
)
```

### Custom Column Prefix

```python
# Use custom prefix for column names
df_with_eb = add_ebb_estimate(
    df,
    success_col='hits',
    total_col='at_bats',
    prefix='bayes'
)

# Columns will be: bayes_raw, bayes_fitted, bayes_low, bayes_high
```

### Access Fitted Prior

```python
df_with_eb = add_ebb_estimate(df, success_col='hits', total_col='at_bats')

# Access the prior stored in DataFrame attributes
prior = df_with_eb.attrs['ebb_prior']
print(f"Prior mean: {prior.mean:.3f}")
```

## See Also

- [`fit_beta_prior()`](fit_beta_prior.html) - Fit beta prior to data
- [`posterior_parameters()`](posterior_parameters.html) - Compute posterior parameters
- [`ebb_prop_test()`](ebb_prop_test.html) - Empirical Bayes hypothesis testing

## References

Ported from David Robinson's R `ebbr` package: https://cran.r-project.org/package=ebbr
