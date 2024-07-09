import numpy as np
import pandas as pd 
from numba import jit

from lib.correlation import _check_arg, cross_correlation
from lib.misc import haversine_distance, import_dataset
from lib.bayes import (posterior_link_probability_havlin,
                       posterior_link_probability_iaaft)

from netCDF4 import Dataset


import multiprocessing as mp
from itertools import product
#############################



def save_results(i,j,Z,the_lagmax,prob,foutput):
    with open(foutput,'a') as file:
        file.write(f"{i}\t{j}\t{Z:.4f}\t{the_lagmax}\t{prob:.4f}\n")


def correlation_all(data,data_surr,foutput):
    T,N = data.shape

    # Rescale the series to return a normalized cross-correlation
    for i in range(0,N):
        data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()
        data[:,i] = data[:,i]/np.sqrt(N) # This is to return a normalized cross-correlation

    
    for i in range(0,N):
        print(f"Computing node {i}")
        for j in range(i+1,N):
            # print(j)
            dist = haversine_distance( nodes[i][0],nodes[i][1], nodes[j][0],nodes[j][1])
            x  = data[:,i]
            y  = data[:,j]
            surr_x = data_surr[:,:,ind_nodes[i][0],ind_nodes[i][1]]
            surr_y = data_surr[:,:,ind_nodes[j][0],ind_nodes[j][1]]

            Z, prob,crossmax,the_lagmax = posterior_link_probability_iaaft(x,y,surr_x,surr_y,
                                                                           dist,max_lag,num_surr=num_surr)
            # Z, prob,crossmax,the_lagmax = posterior_link_probability_havlin(cross_corr,dist,max_lag)
            if prob > 1e-2:
                save_results(i,j,Z,the_lagmax,prob,foutput)

            # print(f"Computing nodes {i} and {j}: corrmax {crossmax:.4f} at lag {the_lagmax}, prob {prob:.4f}")


#############################



if __name__ == "__main__":

    # Parameters
    size = 5    # Size of the grid in degree

    # Input folder surrogates
    # finputsurr = "./IAAFT_surrogates/All/surr_IAAFT_t2m_1970_2022.nc"
    finputsurr = "./IAAFT_surrogates/All/surr_IAAFT_t2m_2022_2100_highemission.nc"
    data_surr_all = Dataset(finputsurr,"r")

    # Load data
    fileinput = f'../data/t2m/t2m_tas_projections_2022_2100.nc'
    # fileinput = f'../data/t2m/anomalies_t2m_1970_2022_5grid.nc'
    
    variable = fileinput.split("_")[1] # t2m tp total_precipitation
    data, indices, nodes, ind_nodes = import_dataset(fileinput,variable)

    max_lag = 150
    num_surr = 30
    # years   = range(1970,2022)  # from 1970 to 2022
    years   = range(2022,2100)  # from 1970 to 2022

    pool = mp.Pool(5)   # Use the number of cores of your PC
    
    for y,year in enumerate(years):
        foutput = f'./Output/correlations/{variable}_year_{years[y]}_maxlag_{max_lag}.csv'    
        
        # Read surrogates
        data_surr = np.array(data_surr_all['t2m'][0:num_surr,indices[y]:indices[y+1],:,:])

        # correlation_all(data[indices[y]:indices[y+1],:],data_surr,foutput)  # Uncomment to not parallelize
        pool.apply_async(correlation_all, args = (data[indices[y]:indices[y+1],:], data_surr,foutput )) # Parallelize
    pool.close()
    pool.join()




