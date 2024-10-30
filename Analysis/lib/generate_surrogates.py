import numpy as np
from netCDF4 import Dataset
from lib.misc import extract_year_limits
import lib.iaaft as iaaft
from datetime import datetime, timedelta
import os

def create_surrogates(fileinput: str,
                      num_surr: int,
                      var_name: str,
                      time_periods_limits: np.array,
                      foutput: str):
    """
    Parameters
    ----------
    fileinput : str
        DESCRIPTION.
    num_surr : int
        DESCRIPTION.
    var_name : str
        DESCRIPTION.
    time_periods_limits : np.array
        DESCRIPTION.
    foutput : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    data = Dataset(fileinput, 'r')    

    print("\n\n Initialize surrogates dataset\n\n")

    # Create dataset to store surrogates
    nlats = data.dimensions['latitude'].size
    nlons = data.dimensions['longitude'].size
    ntimes = data["time"].size
    
    if os.path.exists(foutput):
        os.system(f"rm {foutput}")
    
    surr_dataset = Dataset(foutput,"w")


    surr_dataset.createDimension("lat",nlats)
    surr_dataset.createDimension("lon",nlons)
    surr_dataset.createDimension("time",ntimes)
    surr_dataset.createDimension("surrogate",num_surr)

    # IMPORTANT: setting the chunksizes is crucial for speed
    surr_dataset.createVariable(var_name,float,('surrogate','time','lat','lon'),
                                chunksizes=(num_surr,ntimes,1,1))
    
    print("\n\n Populate surrogates dataset\n\n")
    # Here we create and save IAAFT surrogates for each year
    
    # years   = range(1970,2022)  # from 1970 to 2022

    # yr_limits = extract_year_limits(data)


    # surr_dataset_arr = np.empty(shape=(num_surr,ntimes,nlats,nlons))

    for i in range(nlats):
        # print(f"Generating surrogates for nodes of lat: {i}")
        for j in range(nlons):
            print(f'Surrogating node ({i},{j}) :')
            
            # fill_dataset(data,surr_dataset,ind_ini,ind_end,ntimes,num_surr,i,j,var_name)
            data_slice = data[var_name][:,i,j]

            surr_temp = np.empty(shape=(num_surr,ntimes),dtype=np.float32)
            for n,idx in enumerate(time_periods_limits[:-1]):
                
                start_id = time_periods_limits[n]+1
                end_id = time_periods_limits[n+1]+1
                
                # print(f"Creating surrogates for year {y}")
                surr = iaaft.surrogates(x = data_slice[start_id:end_id], ns=num_surr, verbose=False)

                for s in range(num_surr):    # Normalize and rescale 
                    surr[s,:] = ((surr[s,:]-surr[s,:].mean())/(surr[s,:].std()*np.sqrt(len(surr[s,:]))))

                surr_temp[:,start_id:end_id] = surr
            
            surr_dataset[var_name][:,:,i,j] = surr_temp

    
    surr_dataset.close()


# D = data['time'].units.split("days since ")[1]
# datetime.strptime(D, '%Y-%m-%d') + timedelta(float(data['time'][0].data))
