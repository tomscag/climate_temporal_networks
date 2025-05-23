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
def posterior_link_probability_iaaft(x,y,surr_x,surr_y,dist,max_lag,num_surr=30):
    '''
        We compute the null model using IAAFT surrogates

        INPUT
            x,y (array):        the original data
            surr_x, surr_y      surrogates of the series x and y
            dist (float):       distance in km between x and y
    '''

    cross_corr = cross_correlation(x, y, max_lag, normalize=False)

    crossmax   = max(np.abs(cross_corr))
    the_lagmax = np.abs(cross_corr).argmax() - (max_lag + 1)


    # surr_x = iaaft.surrogates(x=x, ns= num_surr, verbose=False)
    # surr_y = iaaft.surrogates(x=y, ns= num_surr, verbose=False)

    cross_corr_surr = np.empty(num_surr)

    for n in range(0,num_surr):
        sh_cross_corr = cross_correlation(surr_x[n,:], surr_y[n,:], 
                                          max_lag, normalize=False) # surrogates are already normalized when generated
        sh_crossmax   = sh_cross_corr.max() 
        cross_corr_surr[n] = sh_crossmax

    zscore = abs(crossmax - cross_corr_surr.mean())/cross_corr_surr.std()

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

    return zscore, prob,crossmax,the_lagmax



