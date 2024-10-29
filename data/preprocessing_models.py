#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def extract_netcdf(infolder: str):
    print("\n\n Unzip datasets:\n\n")
    for file in glob.glob("./Datasets/"+infolder+"/*.zip"):
        print(file)
        os.system(f"unzip -q -o {file} -d ./Datasets/{infolder}/")
    os.system(f"rm ./Datasets/{infolder}/*.json")
    os.system(f"rm ./Datasets/{infolder}/*.png")


def regridding(infolder: str):

    print("\n\n Regridding datasets:\n\n")
    for fname_input in glob.glob("./Datasets/"+infolder+"/*.nc"):

        print(fname_input)
        # fname_output = './temp_regridded/' + fname_input.split("r4i1p1f1_gn_")[1][0:4] + '.nc'
        fname_output = fname_input.split(".nc")[0] + "_regridded.nc"

        # netcdf4   cfgrib
        ds = xr.open_dataset(fname_input, engine='netcdf4')
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
        ds_regridded.to_netcdf(fname_output, mode='w', format='NETCDF4')
        ds_regridded.close()

        os.system(f"rm {fname_input}")

        # Set time as record dimension
        os.system(f"ncks -O --mk_rec_dmn time {fname_output} {fname_output}")


def create_day_of_year(infile: str):

    print("\n\n Creating 'day of year' variable:\n\n")
    # To be done once when creating the combined file
    data = Dataset(infile, "r+", clobber=True)
    data.createVariable("dayofyear", "i4", ("time",))
    cal_name = data["time"].calendar

    # Extract start date
    if (cal_name == "365_day") or (cal_name == "noleap"):  # No leap year, as most model are
        start_date = data["time"].units.split("days since")[1].strip()
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        shift_years = int(data["time"][0]//365)
        start_date = start_date.replace(year=start_date.year + shift_years)

        # Create day_of_year vector for 365 calendar
        temp = np.floor(data["time"][:])
        temp = temp - temp[0]
        day_of_year = [i % 365 for i in temp]
        day_of_year = [int(i)+1 for i in day_of_year]
        data['dayofyear'][:] = day_of_year

    elif (cal_name == "proleptic_gregorian"):
        print("Presence of leap years")
        start_date = data["time"].units.split("days since")[1].strip()
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        day_of_year = [(start_date + timedelta(days=item)
                        ).timetuple().tm_yday for item in list(data["time"][:])]
        data['dayofyear'][:] = day_of_year

    else:
        print(f"Calendar {cal_name} (not recognized)")
        exit()

    print(f"Start date: {start_date}")

    data.close()

def compute_anomalies(ds: xr.Dataset, 
                      var_name: str, 
                      baseline: list,
                      standardize_anomalies: bool=False):

    print("\n\n Computing anomalies:\n\n")
    start, end = baseline
    # ds          = ds[var_name]
    ds_baseline = ds.where((ds['time.year'] >= start) & (
        ds['time.year'] <= end), drop=True)

    gb = ds_baseline.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    std_clim = gb.std(dim='time')

    # reindex to full time series
    clim_time = clim.sel(dayofyear=ds.time.dt.dayofyear)
    

    # Assign array to a new dataset
    anomalies = ds
    if standardize_anomalies: # I think it's better to avoid divide by std_dev (since can be very small for eg. precipitation )
        std_clim_time = std_clim.sel(dayofyear=ds.time.dt.dayofyear)
        anomalies[var_name][:] = ((ds - clim_time)/std_clim_time)[var_name] 
    else:
        anomalies[var_name][:] = ((ds - clim_time))[var_name]
    
    # Store attributes
    anomalies = anomalies.assign_attrs(
        climate_variable_name=var_name,
        baseline=baseline,
        standardize_anomalies=int(standardize_anomalies))
    
    return anomalies


if __name__ == "__main__":

    if len(sys.argv) > 1:
        infolder = sys.argv[1]
        print(infolder)
    else:
        print("Insert a input folder")
        infolder = "ssp2_4_5_awi_cm_1_1_mr_near_surface_air_temperature"
        # exit()
    print(os.getcwd())
    os.system(f"rm ./Datasets/{infolder}/*.nc")

    ##############################

    extract_netcdf(infolder)

    regridding(infolder)

    # Concatenate files
    filecombined = str(f"./Datasets/{infolder}.nc")
    os.system(f"ncrcat ./Datasets/{infolder}/*_regridded.nc {filecombined}")
    os.system(f"rm ./Datasets/{infolder}/*_regridded.nc")

    ##############################

    create_day_of_year(str(filecombined))

    ##############################



    data = Dataset(filecombined, "r")
    var_list = list(data.variables)
    if 'pr' in var_list:
        var_name = 'pr'
    elif 'tas' in var_list:
        var_name = 'tas'
    elif 't2m' in var_list:
        var_name = 't2m'
    else:
        print("Variable not recognized")
        exit()
    print(f"Climatological variable: {var_name}")
    if "CMIP6" in filecombined:
        baseline = [2022, 2041]
    elif "HISTOR".lower() in filecombined:
        baseline = [1970, 1989]
    else:
        baseline = [2022, 2041]

    FILENAME_OUTPUT = "./Datasets/anomalies_" + \
        f"{var_name}_" + filecombined.split("Datasets/")[1]
    ds = xr.open_dataset(filecombined, engine='netcdf4')    # netcdf4   cfgrib
    anomalies = compute_anomalies(ds, var_name, baseline)
    anomalies.to_netcdf(FILENAME_OUTPUT)

    print("\n\n FINISHED \n\n")
