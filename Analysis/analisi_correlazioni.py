import numpy as np
import pandas as pd 
from numpy.lib.stride_tricks import as_strided

from lib.correlation import _check_arg, crosscorrelation
from lib.misc import haversine_distance, import_dataset, first_day_of_year_index
from lib import iaaft

import statistics
import math

import multiprocessing as mp

#############################


def prior_link_probability(dist,K=2000):

    # Prior probaility for the null hypothesis
    prior = 1 - math.exp(-dist/K)

    return prior


def posterior_link_probability_iaaft(x,y,cross_corr,dist,max_lag,num_surr=50):
    '''
        We compute the null model using IAAFT surrogates

        INPUT
            x,y (array):        the original data
            cross_corr (array): original cross-correlation
            dist (float):       distance in km between x and y
    '''

    crossmax   = max(abs(cross_corr))
    the_lagmax = abs(cross_corr).argmax() - (max_lag + 1)


    surr_x = iaaft.surrogates(x=x, ns= num_surr, verbose=False)
    surr_y = iaaft.surrogates(x=y, ns= num_surr, verbose=False)

    cross_corr_surr = np.empty(num_surr)

    for n in range(0,num_surr):
        serie1_surr = pd.Series(surr_x[n])
        serie2_surr = pd.Series(surr_y[n])
        sh_cross_corr = crosscorrelation(serie1_surr, serie2_surr, max_lag)
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

    # Posterior probability of link existence
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

def save_results(i,j,Z,the_lagmax,prob,foutput):
    with open(foutput,'a') as file:
        file.write(f"{i}\t{j}\t{Z:.4f}\t{the_lagmax}\t{prob:.4f}\n")
    


def correlation_all(data,foutput):
    T,N = data.shape

    for i in range(1500,N):
        print(f"Computing node {i}")
        for j in range(i+1,N):
            dist = haversine_distance( nodes[i][0],nodes[i][1], nodes[j][0],nodes[j][1])
            x  = data[:,i]
            y  = data[:,j]
            cross_corr = crosscorrelation(x, y, max_lag)
            # crossmax   = cross_corr.max()   # Put abs(cross_corr) to consider negative lags too
            # the_lagmax = cross_corr.argmax() - (max_lag + 1)

            Z, prob,crossmax,the_lagmax = posterior_link_probability_iaaft(x,y,cross_corr,dist,max_lag,num_surr=50)
            # Z, prob,crossmax,the_lagmax = posterior_link_probability_havlin(cross_corr,dist,max_lag)
            if prob > 1e-2:
                save_results(i,j,Z,the_lagmax,prob,foutput)

            # print(f"Computing nodes {i} and {j}: corrmax {crossmax:.4f} at lag {the_lagmax}, prob {prob:.4f}")


#############################



if __name__ == "__main__":

    # Parameters
    varname  = "total_precipitation"
    variable = 'tp' # t2m tp
    size = 5    # Size of the grid in degree

    # Load data
    # fileinput = f'../data/temperature/std_anomalies_temperature_pressure_750_{size}grid.nc'
    fileinput = f'../data/total_precipitation/std_anomalies_total_precipitation_1970_2022_{size}grid.nc'
    data, indices, nodes = import_dataset(fileinput,variable)

    max_lag = 50
    years   = range(1970,2022)  # from 1970 to 2022


    # pool = mp.Pool(8)   # Use the number of cores of your PC

    for year,y in enumerate(years):
        foutput = f'./Output/correlations/{varname}_year_{years[year]}_maxlag_{max_lag}.csv'    
        # pool.apply_async(correlation_all, args = (data[indices[year]:indices[year+1],:], foutput, )) # Parallelize
        correlation_all(data[indices[year]:indices[year+1],:],foutput)  # Uncomment to not parallelize
    # pool.close()
    # pool.join()