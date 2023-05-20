# Read data 
# (select climat_env conda environment)

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyunicorn import climate




def Load_dataset(DATA_FILENAME):

    ds = xr.open_dataset(DATA_FILENAME, engine='netcdf4')    # netcdf4   cfgrib
    return ds


DATA_FILENAME = "t2m/2009.nc"

#  Type of data file ("NetCDF" indicates a NetCDF file with data on a regular lat-lon grid)
FILE_TYPE = "NetCDF"

#  Indicate data source (optional)
# DATA_SOURCE = "ncep_ncar_reanalysis"

#  Name of observable in NetCDF file
OBSERVABLE_NAME = "t2m"

#  Select a subset in time and space from the data (e.g., a particular region
#  or a particular time window, or both)
WINDOW = {"time_min": 0., "time_max": 0., "lat_min": 20., "lon_min": 20.,
          "lat_max": 20., "lon_max": 20.}  # selects the whole data set

#Indicate the length of the annual cycle in the data (e.g., 12 for monthly
#  data). This is used for calculating climatological anomaly values
#  correctly.
TIME_CYCLE = 365

#  Related to climate network construction

#  For setting fixed threshold
THRESHOLD = 0.5

#  Indicates whether to use only data from winter months (DJF) for calculating
#  correlations
WINTER_ONLY = False



#  Create a ClimateData object containing the data and print information

data = climate.ClimateData.Load(
    file_name=DATA_FILENAME, observable_name=OBSERVABLE_NAME,
    file_type=FILE_TYPE,
    window=WINDOW, time_cycle=TIME_CYCLE)

#  Print some information on the data set
print(data)





show_earth    = False
show_tseries  = False






# filter using time
# ds_filtered = ds.where( (ds['time.month'] < 10) & (ds['time.month'] > 4), drop=True)


# Filtering using space
lon_value = 11.0
lat_value = 44.0
ds_filtered = ds.sel(lon=lon_value,lat=lat_value)




fig, ax= plt.subplots(figsize=(18,8))
ax.plot(ds_filtered.time, ds_filtered.t2m - 273.15)


ax.tick_params(axis='both', which='major', labelsize=17)
ax.set_xlabel('Time [day]', fontsize=18)
ax.set_ylabel('T [°K]', fontsize=18)
ax.set_title('latitude = {}, longitude = {}'.format(lat_value, lon_value), fontsize=20)
ax.grid(ls='--')
# plt.savefig('./images/time_series.pdf', dpi=300, facecolor='white')
plt.show()



###################
# print("File imported: {}".format(filename))
# print("Imported variables: {}".format(ds.t2m.long_name))

# var_name = ds.t2m.attrs['long_name']



if show_earth:
    temp = ds.t2m.sel(time='2023-02-01T00:00:00.000000000')
    temp = temp - 273.15   # Farenheit to Celsius
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines(resolution="50m")  # 10m 50m 110m
    plot = temp.plot(cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), cbar_kwargs={"shrink": 0.6})
    # plt.title("ERA5 - 2m temperature January 2021")
    plt.title("ERA "+var_name)


if show_tseries:
    long = 12.00
    lat  = [ 45.5]
    temp = ds.t2m.sel(longitude=long,latitude=lat)
    temp = temp - 273.15   # Farenheit to Celsius
    fig = plt.figure(figsize=(10, 10))
    temp.plot.line(x="time")

    temp1 = ds.t2m.sel(longitude=37.75,latitude=55.75)
    temp1 = temp1 - 273.15
    temp1.plot.line(x="time")
    plt.ylabel(var_name,fontsize=18)
    plt.xlabel("time",fontsize=18)
    plt.title("ERA "+var_name) #+' - '+" Lat "+str(lat)+ " Lon " + str(long),fontsize = 18)
    plt.grid(True)
    plt.legend(["Padua","Moscow"],fontsize=20)

# plt.savefig(filename+'.png')
# plt.contourf(ds_z1000['t2m'][1])
# plt.colorbar()
plt.show()