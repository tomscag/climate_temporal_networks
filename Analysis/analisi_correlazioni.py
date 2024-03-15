import numpy as np
import pandas as pd 


from lib.correlation import _check_arg, cross_correlation
from lib.misc import haversine_distance, import_dataset
from lib.bayes import (posterior_link_probability_havlin,
                       posterior_link_probability_iaaft)

import statistics


import multiprocessing as mp

#############################



def save_results(i,j,Z,the_lagmax,prob,foutput):
    with open(foutput,'a') as file:
        file.write(f"{i}\t{j}\t{Z:.4f}\t{the_lagmax}\t{prob:.4f}\n")
    


def correlation_all(data,foutput):
    T,N = data.shape

    for i in range(0,N):
        print(f"Computing node {i}")
        for j in range(i+1,N):
            dist = haversine_distance( nodes[i][0],nodes[i][1], nodes[j][0],nodes[j][1])
            x  = data[:,i]
            y  = data[:,j]
            cross_corr = cross_correlation(x, y, max_lag)
            # crossmax   = cross_corr.max()   # Put abs(cross_corr) to consider negative lags too
            # the_lagmax = cross_corr.argmax() - (max_lag + 1)

            # Z, prob,crossmax,the_lagmax = posterior_link_probability_iaaft(x,y,cross_corr,dist,max_lag,num_surr=50)
            Z, prob,crossmax,the_lagmax = posterior_link_probability_havlin(cross_corr,dist,max_lag)
            if prob > 1e-2:
                save_results(i,j,Z,the_lagmax,prob,foutput)

            # print(f"Computing nodes {i} and {j}: corrmax {crossmax:.4f} at lag {the_lagmax}, prob {prob:.4f}")


#############################



if __name__ == "__main__":

    # Parameters
    size = 5    # Size of the grid in degree

    # Load data
    # fileinput = f'../data/temperature/std_anomalies_temperature_pressure_750_{size}grid.nc'
    # fileinput = f'../data/t2m/anomalies_t2m_1970_2022_5grid.nc'
    fileinput = f'../data/t2m/t2m_tas_projections_2022_2100.nc'
    variable = fileinput.split("_")[1] # t2m tp total_precipitation
    data, indices, nodes = import_dataset(fileinput,variable)

    max_lag = 150
    # years   = range(1970,2022)  # from 1970 to 2022
    years   = range(2022,2100)  # from 1970 to 2022

    pool = mp.Pool(8)   # Use the number of cores of your PC

    for year,y in enumerate(years):
        foutput = f'./Output/correlations/{variable}_year_{years[year]}_maxlag_{max_lag}.csv'    
        pool.apply_async(correlation_all, args = (data[indices[year]:indices[year+1],:], foutput, )) # Parallelize
        # correlation_all(data[indices[year]:indices[year+1],:],foutput)  # Uncomment to not parallelize
    pool.close()
    pool.join()