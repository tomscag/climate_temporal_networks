from lib.generate_surrogates import create_surrogates
import os
import h5py
import multiprocessing as mp
from netCDF4 import Dataset
from lib.bayes import posterior_link_probability_iaaft
from lib.misc import haversine_distance, import_dataset
import numpy as np
import traceback
import sys

# Control the number of threads per core (fixed to 1 to avoid clash with mutliprocessing)
from os import environ
environ['OMP_NUM_THREADS'] = '1'


#############################


def create_hdf_dataset(fnameout):
    fout = h5py.File(fnameout, "w")
    fout.create_dataset("results", shape=(2664, 2664, 3), dtype="f")
    fout.attrs['finput'] = str(fileinput)
    # fout.attrs['finputsurr'] = str(finputsurr)
    fout.close()


def save_results(fout, i, j, Z, the_lagmax, prob):
    fout["results"][i, j, 0] = Z
    fout["results"][i, j, 1] = the_lagmax
    fout["results"][i, j, 2] = prob

def error_callback(error):
    print(f"Error: {error}")
    # Print the full stack trace
    traceback.print_exc()

def result_callback(result):
    print(f"Result: {result}")

def correlation_all(data, foutput_yrs, fnameout):
    
    try:
        # Read surrogates
        data_surr = np.array(
            Dataset(foutput_yrs, "r")[var_name])
        
        T, N = data.shape
        fout = create_hdf_dataset(fnameout)
        # Rescale the series to return a normalized cross-correlation
        for i in range(0, N):
            data[:, i] = (data[:, i]-data[:, i].mean())/data[:, i].std()
            # This is to return a normalized cross-correlation
            data[:, i] = data[:, i]/np.sqrt(T)
    
        with h5py.File(fnameout, mode='r+') as fout:
            for i in range(0, N):
                print(f"Computing node {i}")
                sys.stdout.flush()
                for j in range(i+1, N):
                    dist = haversine_distance(
                        nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1])
                    x = data[:, i]
                    y = data[:, j]
                    surr_x = data_surr[:, :, ind_nodes[i][0], ind_nodes[i][1]]
                    surr_y = data_surr[:, :, ind_nodes[j][0], ind_nodes[j][1]]
        
                    Z, prob, crossmax, the_lagmax = posterior_link_probability_iaaft(x, y, surr_x, surr_y,
                                                                                     dist, max_lag, num_surr=num_surr)
                    save_results(fout, i, j, Z, the_lagmax, prob)
        return True
    
    except Exception as exc:
        print("ERROR!!!!")
        return f"Error process {exc}"
#############################


if __name__ == "__main__":
    
    # Use the number of cores of your PC
    num_cpus = 15
    
    # Save results in
    outfolder = "../Output"
    
    # Load data
    infolder = '/mnt/'
    fileinput = 'anomalies_tas_ssp5_8.5_model_awi_cm_1_1_mr.nc' 
    infilepath = infolder + fileinput
    # fileinput = "/mnt/era5_t2m_1970_2020_anomalies.nc"

    var_name = fileinput.split("_")[1]  # t2m tp total_precipitation
    data, indices, nodes, ind_nodes = import_dataset(infilepath, var_name)

    max_lag = 150
    num_surr = 100
    #years = range(1970, 2021)  # from 1970 to 2020
    years = range(2022,2101)  # from 2022 to 2100

    # Create surrogates
    foldersurr = f"/mnt/surrogates/surr_{fileinput.strip('.nc')}_nsurr_{num_surr}/"
    if not os.path.exists(foldersurr):  # Create the folder if not exists
        os.makedirs(foldersurr)
        create_surrogates(infilepath, num_surr, var_name, indices, years, foldersurr)
    else:
        print("Surrogates directory found!")

    with mp.Pool(processes=num_cpus) as pool:
        for y, year in enumerate(years):
    
            # print(year)
            fnameout = f'{outfolder}/{var_name}_year_{year}_maxlag_{max_lag}.hdf5'
    
            # Read surrogates
            foutput_yrs = (foldersurr + 
                           foldersurr.split("surr_")[1].strip("/").split(".nc")[0] 
                           + '_' + str(year) + '.nc')
    
    
            #correlation_all(data[indices[y]:indices[y+1],:],foutput_yrs,fnameout)  # Uncomment to not parallelize
            pool.apply_async(correlation_all, 
                             args=(data[indices[y]:indices[y+1], :], 
                                   foutput_yrs, fnameout),
                             callback=result_callback,
                             error_callback=error_callback
                             )  # Parallelize
        
        pool.close()
        pool.join()
