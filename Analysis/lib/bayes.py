import numpy as np
import pandas as pd
import math
from lib import iaaft
from lib.correlation import cross_correlation
from numba import jit

# @jit(nopython=True)
def prior_link_probability(dist,K=2000):

    # Prior probaility for the null hypothesis
    prior = 1 - math.exp(-dist/K)

    return prior

def prior_global_null_prob(altern=0.20):
    # altern: global prior for alternative hypothesis
    # Return the prior probaility for the null hypothesis
    return 1 - altern


def safe_log(x):
    """Compute the natural logarithm of x, or return NaN if x is invalid."""
    try:
        return math.log(x)
    except ValueError:  # Catch domain errors (e.g., log(0) or log(negative))
        return float('nan')

#@jit(nopython=True)
def posterior_link_probability_iaaft(
    x: np.ndarray,
    y: np.ndarray,
    surr_x: np.ndarray,
    surr_y: np.ndarray,
    dist: float,
    max_lag: int,
    num_surr: int = 30
):
    """
    Compute the posterior probability of a causal link between two time series
    using cross-correlation and an IAAFT surrogate-based null model.

    The method compares the maximum absolute cross-correlation of the observed
    signals to the distribution obtained from surrogate pairs, then transforms
    this evidence into a posterior link probability that incorporates a prior
    based on spatial distance.

    Parameters
    ----------
    x, y : np.ndarray
        Original one-dimensional time series of equal length.
    surr_x, surr_y : np.ndarray
        Arrays of IAAFT surrogates for `x` and `y`, respectively.
        Expected shapes: (num_surr, len(x)).
    dist : float
        Spatial distance (e.g., in kilometers) between the two series.
    max_lag : int
        Maximum lag (in samples) for cross-correlation computation.
    num_surr : int, optional
        Number of surrogate pairs (default = 30).

    Returns
    -------
    zscore : float
        Z-score of the observed correlation peak relative to surrogate distribution.
    prob : float
        Posterior probability of a link between `x` and `y`.
    crossmax : float
        Maximum absolute cross-correlation value from the original data.
    peak_lag : int
        Lag (in samples) corresponding to `crossmax`.

    Notes
    -----
    - Surrogates are assumed to be precomputed and normalized.
    - The cross-correlation is computed without normalization
      (use `normalize=False` in `cross_correlation`).
    - The prior link probability is distance-dependent, computed via
      `prior_link_probability(dist, K)` with K=2000.
    - The evidence function follows a soft thresholding rule based on the
      complementary error function.

    References
    ----------
    Schreiber & Schmitz (2000), "Surrogate time series," *Physica D* 142(3–4):346–382.

    See also
    --------
    cross_correlation : computes lag-limited cross-correlation.
    prior_link_probability : computes spatial prior probability.
    """

    # --- Compute empirical cross-correlation and its peak
    cross_corr = cross_correlation(x, y, max_lag, normalize=False)
    crossmax   = max(np.abs(cross_corr))
    peak_lag = np.abs(cross_corr).argmax() - max_lag 



    # --- Compute surrogate cross-correlation maxima
    cross_corr_surr = np.empty(num_surr)
    for n in range(0,num_surr):
        sh_cross_corr = cross_correlation(surr_x[n,:], surr_y[n,:], 
                                          max_lag, normalize=False) # surrogates are already normalized when generated
        sh_crossmax   = sh_cross_corr.max() 
        cross_corr_surr[n] = sh_crossmax

    # --- Compute z-score under surrogate null model
    zscore = abs(crossmax - cross_corr_surr.mean())/cross_corr_surr.std()

    # --- Compute p-value via complementary error function
    pval = math.erfc(zscore)
    
    if pval < math.e**(-1):
        B_value = -math.e*pval*safe_log(pval)
    else:
        B_value = 1
    
    # Prior probaility for the null hypothesis
    K = 2000
    prior = prior_link_probability(dist,K)
    # prior = prior_global_null_prob(altern=0.20)

    # Posterior probability of link existence
    try:
        prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)
    except (OverflowError, ZeroDivisionError) as e:
        #print(f"Exception {e} {B_value} {pval}")
        prob = float('nan')

    return zscore, prob, crossmax, peak_lag



