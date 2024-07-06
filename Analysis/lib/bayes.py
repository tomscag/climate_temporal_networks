import numpy as np
import pandas as pd
import math
from lib import iaaft
from lib.correlation import cross_correlation
from numba import jit

@jit(nopython=True)
def prior_link_probability(dist,K=2000):

    # Prior probaility for the null hypothesis
    prior = 1 - math.exp(-dist/K)

    return prior

def prior_global_null_prob(altern=0.20):
    # altern: global prior for alternative hypothesis
    # Return the prior probaility for the null hypothesis
    return 1 - altern

@jit(nopython=True)
def posterior_link_probability_iaaft(x,y,surr_x,surr_y,dist,max_lag,num_surr=30):
    '''
        We compute the null model using IAAFT surrogates

        INPUT
            x,y (array):        the original data
            surr_x, surr_y      surrogates of the series x and y
            dist (float):       distance in km between x and y
    '''

    cross_corr = cross_correlation(x, y, max_lag)

    crossmax   = max(np.abs(cross_corr))
    the_lagmax = np.abs(cross_corr).argmax() - (max_lag + 1)


    # surr_x = iaaft.surrogates(x=x, ns= num_surr, verbose=False)
    # surr_y = iaaft.surrogates(x=y, ns= num_surr, verbose=False)

    cross_corr_surr = np.empty(num_surr)

    for n in range(0,num_surr):
        serie1_surr = surr_x[n,:]
        serie2_surr = surr_y[n,:]
        sh_cross_corr = cross_correlation(serie1_surr, serie2_surr, max_lag)
        sh_crossmax   = sh_cross_corr.max() 
        cross_corr_surr[n] = sh_crossmax


    zscore = abs(crossmax - cross_corr_surr.mean())/cross_corr_surr.std()

    if zscore < 1:
        pval =1
    else:
        pval = 1/(zscore**2)
    
    if pval < math.e**(-1):
        B_value = -math.e*pval*math.log(abs(pval))
    else:
        B_value = 1
    
    # Prior probaility for the null hypothesis
    K = 2000
    prior = prior_link_probability(dist,K)
    # prior = prior_global_null_prob(altern=0.20)

    # Posterior probability of link existence
    if prior > 1e-9:
        prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)
    else:
        prior = 1e-9
        prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)

    return zscore, prob,crossmax,the_lagmax


def posterior_link_probability_havlin(cross_corr,dist,max_lag):
    '''
        We compute the null model as explained in "Stability of Climate Networks with Time"
        https://doi.org/10.1038/srep00666
    '''
    crossmax   = max(abs(cross_corr))
    the_lagmax = abs(cross_corr).argmax() - (max_lag + 1)

    zscore = (max(abs(cross_corr)) - cross_corr.mean() )/ cross_corr.std()


    if zscore < 1:
        pval =1
    else:
        pval = 1/(zscore**2)
    
    if pval < math.e**(-1):
        B_value = -math.e*pval*math.log(abs(pval))
    else:
        B_value = 1
    
    # Prior probaility for the null hypothesis
    K = 2000
    prior = prior_link_probability(dist,K)

    # Posterior probability of link existence
    prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)

    return zscore, prob,crossmax,the_lagmax