import numpy as np
import pandas as pd 


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
    


def correlation_all(data,foutput,year):
    T,N = data.shape

    # Read surrogates
    data_surr = Dataset(f"./IAAFT_surrogates/IAAFT_surrogates_year_{year}.nc","r+")
    
    # Normalize series
    for i in range(0,N):
        data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()
        data[:,i] = data[:,i]/np.sqrt(N) # This is to return a normalized cross-correlation
    # Normalize surrogates
    # for i,j in product(range(data_surr.dimensions['lat'].size),
    #                    range(data_surr.dimensions['lon'].size)):
    #     print(i,j)
    #     for s in range(30): # surrogates
    #         surr = data_surr['t2m'][s,:,i,j]
    #         data_surr['t2m'][s,:,i,j] = ((surr-surr.mean())/(surr.std()*np.sqrt(N)))


    for i in range(0,N):
        # print(f"Computing node {i}")
        for j in range(i+1,N):
            print(j)
            dist = haversine_distance( nodes[i][0],nodes[i][1], nodes[j][0],nodes[j][1])
            x  = data[:,i]
            y  = data[:,j]
            surr_x = data_surr['t2m'][:,:,ind_nodes[i][0],ind_nodes[i][1]]
            surr_y = data_surr['t2m'][:,:,ind_nodes[j][0],ind_nodes[j][1]]

            Z, prob,crossmax,the_lagmax = posterior_link_probability_iaaft(x,y,surr_x,surr_y,
                                                                           dist,max_lag,num_surr=30)
            # Z, prob,crossmax,the_lagmax = posterior_link_probability_havlin(cross_corr,dist,max_lag)
            if prob > 1e-2:
                save_results(i,j,Z,the_lagmax,prob,foutput)

            # print(f"Computing nodes {i} and {j}: corrmax {crossmax:.4f} at lag {the_lagmax}, prob {prob:.4f}")


#############################



if __name__ == "__main__":

    # Parameters
    size = 5    # Size of the grid in degree

    # Load data
    # fileinput = f'../data/temperature/std_anomalies_temperature_pressure_750_{size}grid.nc'
    fileinput = f'../data/t2m/anomalies_t2m_1970_2022_5grid.nc'
    # fileinput = f'../data/t2m/t2m_tas_projections_2022_2100.nc'
    variable = fileinput.split("_")[1] # t2m tp total_precipitation
    data, indices, nodes, ind_nodes = import_dataset(fileinput,variable)

    max_lag = 150
    years   = range(1970,2022)  # from 1970 to 2022
    # years   = range(1970,1972)  # from 1970 to 2022
    # years   = range(2022,2100)  # from 1970 to 2022

    pool = mp.Pool(20)   # Use the number of cores of your PC

    for y,year in enumerate(years):
        foutput = f'./Output/correlations/{variable}_year_{years[y]}_maxlag_{max_lag}.csv'    
        # pool.apply_async(correlation_all, args = (data[indices[y]:indices[y+1],:], foutput,year )) # Parallelize
        correlation_all(data[indices[y]:indices[y+1],:],foutput,year)  # Uncomment to not parallelize
    pool.close()
    pool.join()



# import pstats
# import cProfile
# if __name__ == "__main__":


#     with cProfile.Profile() as profile:
#         # Parameters
#         size = 5    # Size of the grid in degree

#         # Load data
#         # fileinput = f'../data/temperature/std_anomalies_temperature_pressure_750_{size}grid.nc'
#         fileinput = f'../data/t2m/anomalies_t2m_1970_2022_5grid.nc'
#         # fileinput = f'../data/t2m/t2m_tas_projections_2022_2100.nc'
#         variable = fileinput.split("_")[1] # t2m tp total_precipitation
#         data, indices, nodes, ind_nodes = import_dataset(fileinput,variable)

#         max_lag = 150
#         years   = range(1970,2022)  # from 1970 to 2022
#         years   = range(1970,1971)  # from 1970 to 2022
#         # years   = range(2022,2100)  # from 1970 to 2022

#         pool = mp.Pool(20)   # Use the number of cores of your PC

#         for y,year in enumerate(years):
#             foutput = f'./Output/correlations/{variable}_year_{years[y]}_maxlag_{max_lag}.csv'    
#             # pool.apply_async(correlation_all, args = (data[indices[y]:indices[y+1],:], foutput,year )) # Parallelize
#             correlation_all(data[indices[y]:indices[y+1],:],foutput,year)  # Uncomment to not parallelize
#         pool.close()
#         pool.join()
    
#     results = pstats.Stats(profile)
#     results.sort_stats(pstats.SortKey.TIME)
#     results.print_stats()
#     results.dump_stats("results.prof")
