
import os
# os.system("conda activate climate_env")

import numpy as np

import xarray as xr
import xesmf as xe
import numpy as np
import glob

from datetime import datetime, timedelta
from netCDF4 import Dataset

import sys

##############################
##############################
##############################


if __name__== "__main__":

    if len(sys.argv) > 1:   
        inputfolder = sys.argv[1]
        print(inputfolder)
    else:
        print("Insert a input folder")
        inputfolder = "test_folder"
        # exit()
    print(os.getcwd())
    os.system(f"rm ./Datasets/{inputfolder}/*.nc")

    ##############################
    ##############################

    print("\n\n Unzip datasets:\n\n")
    for file in glob.glob("./Datasets/"+inputfolder+"/*.zip"):
        print(file)
        os.system(f"unzip -q -o {file} -d ./Datasets/{inputfolder}/")
    os.system(f"rm ./Datasets/{inputfolder}/*.json")
    os.system(f"rm ./Datasets/{inputfolder}/*.png")
    
    ##############################
    ##############################

    print("\n\n Regridding datasets:\n\n")
    for fname_input in glob.glob("./Datasets/"+inputfolder+"/*.nc"):
        
        print(fname_input)
        #fname_output = './temp_regridded/' + fname_input.split("r4i1p1f1_gn_")[1][0:4] + '.nc'
        fname_output = fname_input.split(".nc")[0] + "_regridded.nc"

        ds = xr.open_dataset(fname_input, engine='netcdf4')    # netcdf4   cfgrib
        ds_target = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 90+5.0, 5.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0, 360, 5.0), {"units": "degrees_east"}),
        }
        )  
        # define the regridder object (from our source dataarray to the target)
        regridder = xe.Regridder(
            ds, ds_target, "bilinear", periodic=True
        )  # this takes some time to calculate a weight matrix for the regridding
        # regridder

        # now we can apply the regridder to our data
        ds_regridded = regridder(ds)  
        
        # Now we save the regridded data in a netcdf format
        ds_regridded.to_netcdf(fname_output,mode='w',format='NETCDF4')
        ds_regridded.close()

        os.system(f"rm {fname_input}")

        # Set time as record dimension  
        os.system(f"ncks -O --mk_rec_dmn time {fname_output} {fname_output}")
    

    # filecombined = f"./Datasets/pr_CMIP6.nc"    
    filecombined = str(f"./Datasets/{inputfolder}.nc")
    os.system(f"ncrcat ./Datasets/{inputfolder}/*_regridded.nc {filecombined}")
    os.system(f"rm ./Datasets/{inputfolder}/*_regridded.nc")

    ##############################
    ##############################

    print("\n\n Creating 'day of year' variable:\n\n")

    def create_day_of_year(fileinput="../t2m/t2m_tas_projections_2022_2100.nc"):
        # To be done once when creating the combined file
        data = Dataset(fileinput,"r+",clobber=True)
        data.createVariable("dayofyear","i4",("time",))

        # Extract start date
        if data["time"].calendar == "365_day":
            start_date = data["time"].units.split("days since")[1].strip()
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            shift_years = int(data["time"][0]//365)
            start_date = start_date.replace(year=start_date.year + shift_years )
            # start_date = datetime(2022,1,1)
        else:
            print("Calendar not 365 days (not recognized)")
        print(f"Start date: {start_date}")
        
        date_vec = [start_date + timedelta(days=i) for i in range(len(data['time']))]
        day_of_year = [item.timetuple().tm_yday for item in date_vec ]
        data['dayofyear'][:] = day_of_year
        data.close()
        
        
    ##############################


    create_day_of_year(str(filecombined))

    ##############################
    ##############################

    print("\n\n Computing anomalies:\n\n")

    def compute_anomalies(ds,VARIABLE,BASELINE_INTERVAL):

        start, end  = BASELINE_INTERVAL
        # ds          = ds[VARIABLE]
        ds_baseline = ds.where( (ds['time.year'] >= start) & (ds['time.year'] <= end), drop=True)

        gb = ds_baseline.groupby('time.dayofyear')
        clim = gb.mean(dim='time')
        std_clim = gb.std(dim='time')

        # reindex to full time series
        clim_time = clim.sel(dayofyear=ds.time.dt.dayofyear)
        std_clim_time = std_clim.sel(dayofyear=ds.time.dt.dayofyear)
        
        # Assign array to a new dataset
        anomalies = ds
        # anomalies[VARIABLE][:] = ((ds - clim_time)/std_clim_time)[VARIABLE] # I think it's better to avoid divide by std_dev (that can be very small for eg. precipitation )
        anomalies[VARIABLE][:] = ((ds - clim_time))[VARIABLE]
        return anomalies
    

    data = Dataset(filecombined,"r")
    var_list = list(data.variables)
    if 'pr' in var_list:
        VARIABLE = 'pr'
    elif 'tas' in var_list:
        VARIABLE = 'tas'
    elif 't2m' in var_list:
        VARIABLE = 't2m'
    else:
        print("Variable not recognized")
        exit()
    print(f"Climatological variable: {VARIABLE}")
    if "CMIP6" in filecombined:
        BASELINE_INTERVAL = [2022,2041]
    elif "HISTOR".lower() in filecombined:
        BASELINE_INTERVAL = [1970,1989]
    else:
        BASELINE_INTERVAL = [2022,2041]

    FILENAME_OUTPUT = filecombined.split(".nc")[0] + "_anomalies.nc"
    ds = xr.open_dataset(filecombined, engine='netcdf4')    # netcdf4   cfgrib
    anomalies = compute_anomalies(ds,VARIABLE,BASELINE_INTERVAL)
    anomalies.to_netcdf(FILENAME_OUTPUT)


    print("\n\n FINISHED \n\n")