### Script to extract anomalies

import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr



def load_dataset(FILENAME_INPUT):
    ''' Load nc data into an xarray '''
    ds = xr.open_dataset(FILENAME_INPUT, engine='netcdf4')    # netcdf4   cfgrib
    return ds


def compute_anomalies(ds,VARIABLE,BASELINE_INTERVAL):

    start, end  = BASELINE_INTERVAL
    ds          = ds[VARIABLE]
    ds_baseline = ds.where( (ds['time.year'] >= start) & (ds['time.year'] <= end), drop=True)

    gb = ds_baseline.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    std_clim = gb.std(dim='time')

    # reindex to full time series
    clim_time = clim.sel(dayofyear=ds.time.dt.dayofyear)
    std_clim_time = std_clim.sel(dayofyear=ds.time.dt.dayofyear)
    
    anomalies   = ((ds - clim_time)/std_clim_time)

    return anomalies


#################################################
#################################################
#################################################

VARIABLE             = 'tas'    # total_precipitation
# FILENAME_INPUT       = f'../{VARIABLE}/{VARIABLE}_1970_2022_5grid.nc'
# FILENAME_OUTPUT      = f'../{VARIABLE}/std_anomalies_{VARIABLE}_1970_2022_5grid.nc'
# FILENAME_INPUT       = f'../temperature/temperature_pressure_500_5grid.nc'
# FILENAME_OUTPUT      = f'../temperature/std_anomalies_temperature_pressure_500_5grid.nc'
FILENAME_INPUT       = f'../t2m/t2m_tas_projections_2022_2100.nc'
FILENAME_OUTPUT      = f'../t2m/t2m_tas_projections_2022_2100_anomalies.nc'


BASELINE_INTERVAL    = [2022,2042]
ds        = load_dataset(FILENAME_INPUT)
anomalies = compute_anomalies(ds,VARIABLE,BASELINE_INTERVAL)


anomalies.to_netcdf(FILENAME_OUTPUT)

#####


# # Extract some series
# lon = 12.0
# lat = 44.0
# a = anomalies.t2m.sel(lon=lon,lat=lat)



# # Test some correlations
# import scipy.stats as ss
# import numpy as np
# start = 2010
# endd  = 2015

# lon1  = 110.0
# lat1  = -40.0
# lon2  = 110.0
# lat2  = -10.0
# ser1 = anomalies.where( (anomalies['time.year']>=start) & (anomalies['time.year']<=endd) , drop=True).sel(lon=lon1,lat=lat1)
# ser2 = anomalies.where( (anomalies['time.year']>=start) & (anomalies['time.year']<=endd) , drop=True).sel(lon=lon2,lat=lat2)


# rho, pval = ss.pearsonr(ser1,ser2)
# print("Pearson correlation: %f" %rho)
# print("P-value: %f" %pval)


# ser1z = ss.zscore(ser1)
# ser2z = ss.zscore(ser2)
# plt.plot(ser1z,linewidth=1.2)
# plt.plot(ser2z,linewidth=1.2)
# plt.show()


# ## Compute cross correlations
# from scipy import signal    

# correlation = signal.correlate(ser1z,ser2z, mode="full",method="direct")
# correlation /= len(ser2z)
# lags = signal.correlation_lags(ser1.size, ser2.size, mode="full")
# lag  = lags[np.argmax(correlation)]


# # Test
# N = 300
# x = np.random.rand(N)
# y = [0]*N
# y[2:] = x[1:-1] 
# correlation = signal.correlate(x,y, mode="full",method="direct")

# # ''' Python only implementation '''

# # # Pre-allocate correlation array
# # corr = (len(ser1) - len(ser2) + 1) * [0]

# # # Go through lag components one-by-one
# # corr = [0,1,2]
# # for l in range(len(corr)):
# #     print(l)
# #     corr[l] = sum([ser1z[i+l] * ser2z[i] for i in range(len(ser2z))])

# # print(corr[0]/len(ser1z))
