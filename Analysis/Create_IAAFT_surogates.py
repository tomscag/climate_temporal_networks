#   Create IAAFT surogates 

import numpy as np
import pandas as pd
import math
from lib import iaaft
from lib.correlation import cross_correlation

from lib.misc import haversine_distance, import_dataset

from netCDF4 import Dataset



def create_destination_dataset(data,foutput,ntimes):

    # Create dataset to store surrogates
    data_surr = Dataset(foutput,"w")
    nlats = data.dimensions['lat'].size
    nlons = data.dimensions['lon'].size


    # # Copy dimensions
    # for dim_name, dim in data.dimensions.items():
    #     data_surr.createDimension(dim_name, (len(dim) if not dim.isunlimited() else None))

    data_surr.createDimension("lat",nlats)
    data_surr.createDimension("lon",nlons)
    data_surr.createDimension("time",ntimes)
    data_surr.createDimension("surrogate",num_surr)

    data_surr.createVariable("t2m",float,('surrogate','time','lat','lon'))

    return data_surr



def create_IAAFT_surrogates(data):
    # Here we create and save IAAFT surrogates for each year
    

    years   = range(1970,2022)  # from 1970 to 2022

    ind = np.where( np.array(data['dayofyear'])  == 1)[0]
    

    for y,year in enumerate(years):

        print("Creating surrogates for year {y}")
        foutput = f"./IAAFT_surrogates/IAAFT_surrogates_year_{year}.nc"

        ntimes = len(range(ind[y],ind[y+1]))
        data_surr = create_destination_dataset(data,foutput,ntimes)

        num_surr = data_surr.dimensions['surrogate'].size
        nlats = data_surr.dimensions['lat'].size
        nlons = data_surr.dimensions['lon'].size
        

        for i in range(nlats):
            print(f"Generating surrogates for nodes of lat: {i}")
            for j in range(nlons):

                surr = iaaft.surrogates(x = data['t2m'][ind[y]:ind[y+1],i,j], ns= num_surr, verbose=False)
                data_surr['t2m'][:,:,i,j] = surr


        data_surr.close()





if __name__ == "__main__":

    num_surr = 30

    # Load data
    fileinput = f'../data/t2m/anomalies_t2m_1970_2022_5grid.nc'
    foutput = f"./IAAFT_surrogates/"
    # fileinput = f'../data/t2m/t2m_tas_projections_2022_2100.nc'

    # variable = fileinput.split("_")[1] # t2m tp total_precipitation
    # data, indices, nodes = import_dataset(fileinput,variable)

    # Load dataset as netCDF file
    data = Dataset(fileinput,"r")


    max_lag = 150


    create_IAAFT_surrogates(data)  

    # for year,y in enumerate(years):
    #     foutput = f'./IAAFT_surrogates'    
    #     create_IAAFT_surrogates(data[indices[year]:indices[year+1],:],foutput)  # Uncomment to not parallelize



