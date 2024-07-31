#   Create IAAFT surogates 

# To add a function from the parent folder 
# Better solution is here https://stackoverflow.com/a/50194143/11419362
import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import Analysis.lib.iaaft as iaaft # Import from a parent folder
# from ..Analysis.lib.correlation import cross_correlation

# from ..Analysis.lib.misc import haversine_distance, import_dataset

import netCDF4 as nc
import pandas as pd
import multiprocessing as mp

##############################
##############################
##############################


def fill_dataset(data_slice,surr_dataset_arr,ind_ini,ind_end,ntimes,num_surr,i,j):

    surr_temp = np.empty(shape=(num_surr,ntimes),dtype=np.float32)
    for y,id in enumerate(ind_ini):

        # print(f"Creating surrogates for year {y}")
        surr = iaaft.surrogates(x = data_slice[ind_ini[y]:ind_end[y]+1], ns=num_surr, verbose=False)
        # surr = iaaft.surrogates(x = data[var_name][ind_ini[y]:ind_end[y],i,j], ns=num_surr, verbose=False)

        for s in range(num_surr):    # Normalize and rescale 
            surr[s,:] = ((surr[s,:]-surr[s,:].mean())/(surr[s,:].std()*np.sqrt(len(surr[s,:]))))

        surr_temp[:,ind_ini[y]:ind_end[y]+1] = surr

    surr_dataset_arr[:,:,i,j] = surr_temp
    # return surr_temp


def create_IAAFT_surrogates(data,foutput):

    print("\n\n Initialize surrogates dataset\n\n")
    var_name = data.climate_variable_name

    # Create dataset to store surrogates
    surr_dataset = nc.Dataset(foutput,"w")
    nlats = data.dimensions['lat'].size
    nlons = data.dimensions['lon'].size
    ntimes = data["time"].size

    # # Copy dimensions
    # for dim_name, dim in data.dimensions.items():
    #     surr_dataset.createDimension(dim_name, (len(dim) if not dim.isunlimited() else None))

    surr_dataset.createDimension("lat",nlats)
    surr_dataset.createDimension("lon",nlons)
    surr_dataset.createDimension("time",ntimes)
    surr_dataset.createDimension("surrogate",num_surr)

    surr_dataset.createVariable(var_name,float,('surrogate','time','lat','lon'))

    print("\n\n Populate surrogates dataset\n\n")
    # Here we create and save IAAFT surrogates for each year
    
    # years   = range(1970,2022)  # from 1970 to 2022

    ind_ini = np.where( np.array(data['dayofyear'])  == 1)[0]
    ind_end = np.where( np.array(data['dayofyear'])  == 365)[0]
    
    surr_dataset_arr = np.empty(shape=(num_surr,ntimes,nlats,nlons))



    for i in range(nlats):
        # print(f"Generating surrogates for nodes of lat: {i}")
        for j in range(nlons):
            print(f'Surrogating node ({i},{j}) :')
            data_slice = data[var_name][:,i,j]
            fill_dataset(data_slice,surr_dataset_arr,ind_ini,ind_end,ntimes,num_surr,i,j)

    surr_dataset[var_name][:] = surr_dataset_arr
    surr_dataset.close()


    # with mp.Pool(1) as pool:


    #     for i in range(nlats):
    #         # print(f"Generating surrogates for nodes of lat: {i}")
    #         for j in range(nlons):
    #             print(f'Surrogating node ({i},{j}) :')
    #             data_slice = data[var_name][:,i,j]
    #             pool.apply_async(fill_dataset, 
    #                         args = (data_slice,surr_dataset_arr,ind_ini,ind_end,ntimes,num_surr,i,j))

    #             # surr_temp = fill_dataset(data_slice,ind_ini,ind_end,i,j,var_name,ntimes,num_surr)
    #             # surr_dataset[var_name][:,:,i,j] = surr_temp # TODO: this is extremely slow
    #     surr_dataset.close()
    #     pool.close()
    #     pool.join() #TODO: what is the purpose of this???





##############################
##############################
##############################


if __name__ == "__main__":

    num_surr = 30
    filename = "anomalies_pr_CMIP6_ssp5_8.5_model_CESM2.nc"

    # Load data
    fileinput = f'./Datasets/{filename}'
    foutput = f"./surr_{filename}"
    
    
    # Load dataset as netCDF file
    data = nc.Dataset(fileinput,"r")

    # surr_dataset = create_destination_dataset(data,foutput)

    create_IAAFT_surrogates(data,foutput)  


