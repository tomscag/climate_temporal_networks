from os import environ
environ['OMP_NUM_THREADS'] = '1' # Control the number of threads per core (fixed to 1 to avoid clash with mutliprocessing)

import numpy as np

from lib.misc import haversine_distance, import_dataset
from lib.bayes import posterior_link_probability_iaaft

from netCDF4 import Dataset

from lib.generate_surrogates import create_surrogates 

import multiprocessing as mp
import h5py
import os

#############################

def create_hdf_dataset(fnameout):
    fout = h5py.File(fnameout,"w")
    fout.create_dataset("results",shape=(2664,2664,3), dtype="f")
    fout.attrs['finput'] = str(fileinput)
    # fout.attrs['finputsurr'] = str(finputsurr)

    return fout

def save_results(fout,i,j,Z,the_lagmax,prob):
    fout["results"][i,j,0] = Z
    fout["results"][i,j,1] = the_lagmax
    fout["results"][i,j,2] = prob


def correlation_all(data,data_surr,fnameout):
    T,N = data.shape
    fout = create_hdf_dataset(fnameout)

    # Rescale the series to return a normalized cross-correlation
    for i in range(0,N):
        data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()
        data[:,i] = data[:,i]/np.sqrt(T) # This is to return a normalized cross-correlation

    
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
            save_results(fout,i,j,Z,the_lagmax,prob)

#############################


if __name__ == "__main__":


    # Load data
    infolder = '../data/'
    fileinput = 'era5_t2m_1970_2020_anomalies.nc'
    filepath = infolder + fileinput
    # fileinput = "/mnt/era5_t2m_1970_2020_anomalies.nc"
    
    var_name = fileinput.split("_")[1] # t2m tp total_precipitation
    data, indices, nodes, ind_nodes = import_dataset(filepath,var_name)

    max_lag = 150
    num_surr = 30
    years = range(1970,2021)  # from 1970 to 2020
    # years   = range(2022,2101)  # from 2022 to 2100

    # Create surrogates
    fileoutput = f"./surr_{fileinput}"
    if not os.path.exists(fileoutput):  # Create the file if not exists
        create_surrogates(filepath,num_surr,var_name,years,fileoutput)
        data_surr_all = Dataset(fileoutput,"r")
    else:
        data_surr_all = Dataset(fileoutput,"r")


    pool = mp.Pool(8)   # Use the number of cores of your PC

    for y,year in enumerate(years):
        print(year)
        fnameout = f'./Output/{var_name}_year_{years[y]}_maxlag_{max_lag}.hdf5'    
        
        # Read surrogates
        data_surr = np.array(data_surr_all[var_name][0:num_surr,indices[y]:indices[y+1],:,:])

        # correlation_all(data[indices[y]:indices[y+1],:],data_surr,fnameout)  # Uncomment to not parallelize
        pool.apply_async(correlation_all, args = (data[indices[y]:indices[y+1],:], data_surr,fnameout )) # Parallelize
    pool.close()
    pool.join()




