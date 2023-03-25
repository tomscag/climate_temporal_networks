# Import ERA data using API request
# https://cds.climate.copernicus.eu/api-how-to


import cdsapi
c = cdsapi.Client()

catalog   = "reanalysis-era5-single-levels-monthly-means"  # reanalysis-era5-single-levels  reanalysis-era5-single-levels-monthly-means  
type      = "reanalysis"
variable  = "2m_temperature"
year      = "2021"
month     = ["01"]
day       = ["01"]
time      = ["12:00","11:00"]
form      = "grib"   # "grib" "netcdf.zip"
fout      = variable + "_" + year + "." + form


c.retrieve(catalog,
{
'product_type': type,
"variable":     variable,
"pressure_level": "1000",
"product_type": type,
"year":   year,
"month":  month,
"day":    day,
"time":   time,
"format": form
}, fout)


print("Writing file in %s" %(fout))

