"""
Implementation of Nogueira's stability measure. See:
[1] S. Nogueira, K. Sechidis, and G. Brown, “On the Stability of Feature Selection 
 Algorithms,” Journal of Machine Learning Research, vol. 18, no. 174, pp. 1–54, 2018.

"""


from scipy.stats import norm
from collections import namedtuple
from sklearn.utils import check_array
import numpy as np


NogueiraTest = namedtuple(
    "NogueiraTest", ("estimator", "upper", "lower", "var", "error", "alpha")
)


def nogueria_test(pop_mat, alpha=0.05):
    """Let X be a feature space of dimension `n_features` and `pop_mat` a binary matrix 
    of dimension `(n_samples, n_festures)` representing `n_samples` runs of a feature 
    selection algorithm over X (with respect to a response). This function computed the
    Nogueira stability estimate, error, varaince and confindece interval

    Parameters
    ----------
    pop_mat : 2d-array like
        A (n_samples, n_features) binary matrix, each row is a sample of the FS 
        algorithm applied on a n_features space, where a 1 in position (i,j) means that
        the feature j has been selected for the i-th run.

    alpha : scalar
        Level of significance for the CI.

    Returns
    -------
    NogueiraTest
        A named tuple with the results of the stability test.
    """

    pop_mat = check_array(
        pop_mat,
        dtype="numeric",
        ensure_2d=True,
        ensure_min_features=2,
        ensure_min_samples=2,
    )

    (n_samples, n_features) = pop_mat.shape

    # Frequency of selection of each feature
    p_f_hat = np.mean(pop_mat, axis=0)
    # Average number of features selected over the n_samples feature sets
    k_bar = np.sum(p_f_hat)
    # Numer of features selected for each sample
    k = np.sum(pop_mat, axis=1)

    # Sum of unbiased sample variance of each feature
    var_f_sum = (n_samples / (n_samples - 1)) * (p_f_hat * (1 - p_f_hat)).sum()
    num = var_f_sum / n_features
    denom = (k_bar / n_features) * (1 - k_bar / n_features)

    # Estability estimate, eq. (2) in [1]
    estimator = 1 - num / denom

    # Variance of the estimate, Theorem 7 in [1]
    phi = (1 / denom) * (
        np.mean(np.multiply(pop_mat, p_f_hat), axis=1)
        - (k * k_bar) / n_features ** 2
        + (estimator / 2)
        * ((2 * k * k_bar) / n_features ** 2 - k / n_features - k_bar / n_features + 1)
    )
    phi_av = phi.mean()
    phi_var = (4 / n_samples ** 2) * ((phi - phi_av) ** 2).sum()

    # CI of the estimate, Corollary 8 in [1]
    lower = estimator - norm.ppf(1 - alpha / 2) * np.sqrt(phi_var)
    upper = estimator + norm.ppf(1 - alpha / 2) * np.sqrt(phi_var)
    error = estimator - lower

    return NogueiraTest(estimator, upper, lower, phi_var, error, alpha)
