import netCDF4 as nc
import numpy as np
from netCDF4 import Dataset


# COPIARE UN DATASET
''' 
# DATA
data = Dataset('t2m_1970_2022_4grid.nc', 'r')
lat = data.variables['lat']          #len(lat)=91
lon = data.variables['lon']          #len(lon)=180
temp = data.variables['t2m']         #len(temp)=365

print(lat)
print(lon)
print(temp)


Datifiltrati = nc.Dataset('filtered_t2m_1970_2022_4grid.nc', 'w', format='NETCDF4')

# CREAZIONE NUOVO DATASET NETCDF4
for dimname, dim in data.dimensions.items():
    Datifiltrati.createDimension(dimname, len(dim))

lat_nuova = Datifiltrati.createVariable('lat', lat.dtype, lat.dimensions)
lon_nuova = Datifiltrati.createVariable('lon', lon.dtype, lon.dimensions)
temp_nuova = Datifiltrati.createVariable('t2m', temp.dtype, temp.dimensions)

lat_nuova[:] = lat[:]
lon_nuova[:] = lon[:]

#temp_nuova = Datifiltrati.variables['t2m']


print(lat_nuova)
print(lon_nuova)
print(Datifiltrati.variables.keys)
'''


# Create a new NetCDF4 file
nc_filename = '4grid.nc'
nc_file = nc.Dataset(nc_filename, 'w')

# Define the dimensions
decades_dim = nc_file.createDimension('decades', 5)
node_i_dim = nc_file.createDimension('node_i', 4050)
node_j_dim = nc_file.createDimension('node_j', 4050)

# Create the variables
decades_var = nc_file.createVariable('decades', np.int32, ('decades',))
node_i_var = nc_file.createVariable('node_i', np.int32, ('node_i',))
node_j_var = nc_file.createVariable('node_j', np.int32, ('node_j',))
zscoremax = nc_file.createVariable('zscoremax', np.float32, ('decades', 'node_i', 'node_j'))
#zscoremin = nc_file.createVariable('zscoremin', np.float32, ('decades', 'node_i', 'node_j'))
thelagmax = nc_file.createVariable('thelagmax', np.float32, ('decades', 'node_i', 'node_j'))
#thelagmin = nc_file.createVariable('thelagmin', np.float32, ('decades', 'node_i', 'node_j'))
probability = nc_file.createVariable('probability', np.float32, ('decades', 'node_i', 'node_j'))
distance = nc_file.createVariable('distance', np.float32, ('node_i', 'node_j'))
pvalue = nc_file.createVariable('pvalue', np.float32, ('decades', 'node_i', 'node_j'))



# add attributes to variables or dimensions
decades_var.units = 'decades'
node_i_var.units = 'node i'
node_j_var.units = 'node j'
zscoremax.units = 'pure number'
#zscoremin.units = 'pure number'
thelagmax.units = 'days'
#thelagmin.units = 'days'
probability.units = 'pure number'
distance.units = 'pure number'
pvalue.units = 'pure number'
# Close the NetCDF4 file to save it
nc_file.close()
