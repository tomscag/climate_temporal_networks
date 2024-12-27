import numpy as np
from netCDF4 import Dataset
import lib.iaaft as iaaft
from datetime import datetime, timedelta
import os

def create_surrogates(fileinput: str,
                      num_surr: int,
                      var_name: str,
                      date_vec: datetime,
                      years: list,
                      foldersurr: str):
    """
    Parameters
    ----------
    fileinput : str
        DESCRIPTION.
    num_surr : int
        DESCRIPTION.
    var_name : str
        DESCRIPTION.
    date_vec : datetime
        datetime vector
    years: list
        years list.
    foldersurr : str
        Folder where to store the surrogates.

    Returns
    -------
    None.

    """
    
    # Load data to surrogate
    data = Dataset(fileinput, 'r')    
    nlats = data.dimensions['latitude'].size
    nlons = data.dimensions['longitude'].size
    
    print("\n\n Initialize surrogates dataset\n\n")

    for n, year in enumerate(years):
        
        foutput_yrs = (foldersurr + 
                       foldersurr.split("surr_")[1].strip("/").split(".nc")[0] 
                       + '_' + str(year) + '.nc')
        
        ind = [i for i, dt in enumerate(date_vec) if dt.year==year]
        ntimes = len(ind)

        with Dataset(foutput_yrs,"w") as surr_dataset:
            surr_dataset.createDimension("latitude",nlats)
            surr_dataset.createDimension("longitude",nlons)
            surr_dataset.createDimension("time",ntimes)
            surr_dataset.createDimension("surrogate",num_surr)
        
            # IMPORTANT: setting the chunksizes is crucial for speed
            surr_dataset.createVariable(var_name,float,
                                        ('surrogate','time','latitude','longitude'),
                                        chunksizes=(num_surr,ntimes,1,1))
            
            print(f"Creating surrogates for year {years[n]}")
            
            data_slice = data[var_name][ind,:,:]
            
            for i in range(nlats):
                print(f"Surrogating nodes of lat: {i}")
                for j in range(nlons):
                    # print(f'Surrogating node ({i},{j}) :')
                    
                    surr = iaaft.surrogates(x = data_slice[:,i,j], 
                                            ns=num_surr, verbose=False)
            
                    for s in range(num_surr):    # Normalize and rescale 
                        surr[s,:] = ((surr[s,:]-surr[s,:].mean())/(surr[s,:].std()*np.sqrt(ntimes)))
            
                    surr_dataset[var_name][:,:,i,j] = surr
    
