### Script to extract anomalies

import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr

def Plot_anomalies(anomalies,lon=12.0,lat=44.0):

    ser = anomalies.sel(lon=lon,lat=lat)
    ser.plot(figsize=(20,4))
    plt.show()

def Plot_global_avg(anomalies):

    global_avg = anomalies.mean(dim=['lon','lat'])  
    global_avg.plot(figsize=(10,5))
    plt.show()  

def Save_anomalies(anomalies):
    anomalies.to_netcdf('anomalies.nc')

def Load_dataset(DATA_FILENAME):
    ''' Load nc data into an xarray '''
    ds = xr.open_dataset(DATA_FILENAME, engine='netcdf4')    # netcdf4   cfgrib
    return ds


def Compute_anomalies(ds,INTERVAL):
    start_base  = INTERVAL[0]
    end_base    = INTERVAL[1]
    ds          = ds.t2m
    ds_baseline = ds.where( (ds['time.year'] >= start_base) & (ds['time.year'] <= end_base), drop=True)
    climatology = ds_baseline.groupby('time.dayofyear').mean('time')
    anomalies   = ds.groupby('time.dayofyear') - climatology

    return anomalies


########################

DATA_FILENAME = './data/t2m_1970_2022_2grid.nc'
INTERVAL      = [1970,1990]




ds        = Load_dataset(DATA_FILENAME)
anomalies = Compute_anomalies(ds,INTERVAL)



# Extract some series
lon = 12.0
lat = 44.0
a = anomalies.t2m.sel(lon=lon,lat=lat)



# Test some correlations
import scipy.stats as ss
import numpy as np
start = 2010
endd  = 2015

lon1  = 110.0
lat1  = -40.0
lon2  = 110.0
lat2  = -10.0
ser1 = anomalies.where( (anomalies['time.year']>=start) & (anomalies['time.year']<=endd) , drop=True).sel(lon=lon1,lat=lat1)
ser2 = anomalies.where( (anomalies['time.year']>=start) & (anomalies['time.year']<=endd) , drop=True).sel(lon=lon2,lat=lat2)


rho, pval = ss.pearsonr(ser1,ser2)
print("Pearson correlation: %f" %rho)
print("P-value: %f" %pval)


ser1z = ss.zscore(ser1)
ser2z = ss.zscore(ser2)
plt.plot(ser1z,linewidth=1.2)
plt.plot(ser2z,linewidth=1.2)
plt.show()