#   Create IAAFT surogates 

# To add a function from the parent folder 
# Better solution is here https://stackoverflow.com/a/50194143/11419362
import sys
import os 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import Analysis.lib.iaaft as iaaft # Import from a parent folder
from Analysis.lib.misc import extract_year_limits
# from ..Analysis.lib.correlation import cross_correlation

# from ..Analysis.lib.misc import haversine_distance, import_dataset

import netCDF4 as nc
import pandas as pd
import multiprocessing as mp

# from mpi4py import MPI

##############################
##############################
##############################


def create_IAAFT_surrogates(data,foutput):

    print("\n\n Initialize surrogates dataset\n\n")
    var_name = data.climate_variable_name

    # Create dataset to store surrogates
    nlats = data.dimensions['lat'].size
    nlons = data.dimensions['lon'].size
    ntimes = data["time"].size
    surr_dataset = nc.Dataset(foutput,"w")


    surr_dataset.createDimension("lat",nlats)
    surr_dataset.createDimension("lon",nlons)
    surr_dataset.createDimension("time",ntimes)
    surr_dataset.createDimension("surrogate",num_surr)

    # IMPORTANT: setting the chunksizes is crucial for speed
    surr_dataset.createVariable(var_name,float,('surrogate','time','lat','lon'),chunksizes=(num_surr,ntimes,1,1))
    
    print("\n\n Populate surrogates dataset\n\n")
    # Here we create and save IAAFT surrogates for each year
    
    # years   = range(1970,2022)  # from 1970 to 2022

    yr_limits = extract_year_limits(data)

    # surr_dataset_arr = np.empty(shape=(num_surr,ntimes,nlats,nlons))

    for i in range(nlats):
        # print(f"Generating surrogates for nodes of lat: {i}")
        for j in range(nlons):
            print(f'Surrogating node ({i},{j}) :')
            
            # fill_dataset(data,surr_dataset,ind_ini,ind_end,ntimes,num_surr,i,j,var_name)
            data_slice = data[var_name][:,i,j]

            surr_temp = np.empty(shape=(num_surr,ntimes),dtype=np.float32)
            for y,id in enumerate(yr_limits[:-1]):

                # print(f"Creating surrogates for year {y}")
                surr = iaaft.surrogates(x = data_slice[yr_limits[y]:yr_limits[y+1]], ns=num_surr, verbose=False)

                for s in range(num_surr):    # Normalize and rescale 
                    surr[s,:] = ((surr[s,:]-surr[s,:].mean())/(surr[s,:].std()*np.sqrt(len(surr[s,:]))))

                surr_temp[:,yr_limits[y]:yr_limits[y+1]] = surr
            
            surr_dataset[var_name][:,:,i,j] = surr_temp

    
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
    filename = "anomalies_tas_CMIP6_ssp1_2.6_model_CESM2.nc"

    # Load data
    fileinput = f'{filename}'
    foutput = f"./surr_{filename}"
    
    
    # Load dataset as netCDF file
    data = nc.Dataset(fileinput,"r") # parallel=True to enable parallel IO

    # surr_dataset = create_destination_dataset(data,foutput)

    create_IAAFT_surrogates(data,foutput)  


