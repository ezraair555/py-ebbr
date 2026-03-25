import numpy as np
import pandas as pd

from pyebbr import add_ebb_estimate, fit_beta_prior, posterior_parameters


def test_prior_and_estimate():
    data = pd.DataFrame({"success": [3, 8, 40], "total": [10, 40, 120]})
    prior = fit_beta_prior(data["success"], data["total"])
    assert prior.alpha > 0
    alpha1, beta1 = posterior_parameters(data["success"], data["total"], prior)
    assert np.all(alpha1 > data["success"])
    out = add_ebb_estimate(data, success_col="success", total_col="total", prior=prior)
    assert "ebb_fitted" in out.columns
