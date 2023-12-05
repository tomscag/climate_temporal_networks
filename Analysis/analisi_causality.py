# Analisi causalità
import numpy as np
import pandas as pd
import math

from Functions.causality_functions import grangers_causation_matrix
from Functions.other_functions import import_dataset, first_day_of_year_index, haversine_distance

import multiprocessing as mp
from statsmodels.tsa.stattools import grangercausalitytests
#################################



def posterior_link_probability(pval,dist):

    if (pval < math.e**(-1)) & (pval>0):
        B_value = -math.e*pval*math.log(abs(pval))
    elif pval == 0:
        pval += np.nextafter(0,1) # Add a tiny float to avoid 0*log(0)
        B_value = -math.e*pval*math.log(abs(pval))
    else:
        B_value = 1

    # Prior probaility for the null hypothesis
    K = 2000
    prior = 1 - math.exp(-dist/K)

    # Posterior probability of link existence
    prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)

    return  prob

def save_results(i,j,gc,prob,foutput):
    with open(foutput,'a') as file:
        file.write(f"{i}\t{j}\t{gc:.4f}\t{prob:.4f}\n")

def granger_causality_all(data,foutput,test='params_ftest'):
    maxlag = 10
    T,N = data.shape
    for i in range(N):
        print(f"Computing node {i}")
        for j in range(i+1,N):
            if i != j:
                dist = haversine_distance( nodes[i][0],nodes[i][1], nodes[j][0],nodes[j][1])
                x  = data[:,i]
                y  = data[:,j]
                res = grangercausalitytests(np.array([x,y]).T,maxlag=maxlag,verbose=False)
                pvalues = np.array([round(res[i+1][0][test][1],4) for i in range(maxlag)])
                maxind = np.argmax(pvalues)
                pval = pvalues[maxind]
                gc = res[maxind+1][0][test][1]
                prob = posterior_link_probability(pval,dist)
                if prob > 1e-2:
                    save_results(i,j,gc,prob,foutput)

#################################

if __name__ == "__main__":

    # Parameters

    size = 5    # Size of the grid in degree


    # Load data
    fileinput = f'../data/t2m/anomalies_t2m_1970_2022_{size}grid.nc'
    data, indices, nodes = import_dataset(fileinput)


    max_lag = 150
    years   = range(1970,2022)  # from 1970 to 2022


    pool = mp.Pool(8)   # Use the number of cores of your PC

    for year,y in enumerate(years):
        foutput = f'./Output/GC/year_{years[year]}_maxlag_{max_lag}.csv'    
        pool.apply_async(granger_causality_all, args = (data[indices[year]:indices[year+1],:], foutput, )) # Parallelize
        # granger_causality_all(data[indices[year]:indices[year+1],:],foutput)  # Uncomment to not parallelize
    pool.close()
    pool.join()