"""pyebbr – Empirical Bayes binomial estimation utilities."""

from .prior import BetaPrior, fit_beta_prior
from .estimation import add_ebb_estimate, posterior_parameters
from .testing import ebb_prop_test, PropTestResult
from .mixture import BetaMixturePrior, fit_beta_mixture

__all__ = [
    "BetaPrior",
    "BetaMixturePrior",
    "PropTestResult",
    "fit_beta_prior",
    "fit_beta_mixture",
    "posterior_parameters",
    "add_ebb_estimate",
    "ebb_prop_test",
]
