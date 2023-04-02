# Read data 
# (select climat_env conda environment)

import xarray as xr
import cfgrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

show_earth    = True
show_tseries  = False


filename  = "test_1.nc"

ds = xr.open_dataset(filename, engine='netcdf4')    # netcdf4   cfgrib

print("File imported: {}".format(filename))
print("Imported variables: {}".format(ds.t2m.long_name))

var_name = ds.t2m.attrs['long_name']



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