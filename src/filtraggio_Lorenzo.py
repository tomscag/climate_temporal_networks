import netCDF4 as nc
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# DATA
data = Dataset('./data/t2m_1970_2022_2grid.nc', 'r')
lat  = data.variables['lat']          #len(lat)=91
lon  = data.variables['lon']          #len(lon)=180
temp = data.variables['t2m']         #len(temp)=365

'''print(lat)
print(lon)
print(temp)'''

nyear     = 20
nbiyear   = (nyear+1) // 4 
nnormyear = nyear - nbiyear
ndays     = nbiyear * 366 + nnormyear *365
lagRange  = list(range(-365, 365))          #posso usare len(temp[:,i,j])


dailymean1 = [0]*366
dailymean2 = [0]*366
timerange  = list(range(0, 19358))
Nday       = list(range(0, 365))
Nbiday     = list(range(0, 366))
Nyear      = list(range(0, nyear))
Totyear    = list(range(0, 53))
biyear     = [2,6,10,14,18,22,26,30,34,38,42,46,50]
normalyear = [y for y in Totyear if y not in biyear]

Datifiltrati = nc.Dataset('./data/filtered_t2m_1970_2022_2grid.nc', 'r+', format='NETCDF4')

'''for dimname, dim in data.dimensions.items():
    Datifiltrati.createDimension(dimname, len(dim))

lat_nuova = Datifiltrati.createVariable('lat', lat.dtype, lat.dimensions)
lon_nuova = Datifiltrati.createVariable('lon', lon.dtype, lon.dimensions)
lat_nuova[:] = lat[:]
lon_nuova[:] = lon[:]'''

lat = Datifiltrati.variables['lat'][:]          #len(lat)=721
lon = Datifiltrati.variables['lon']
temp_nuova = Datifiltrati.variables['t2m']

'''# Scrivi i dati modificati nella variabile 'temp' nel nuovo file
latN = Datifiltrati.variables['lat'][:]          #len(lat)=721
lonN = Datifiltrati.variables['lon'][:]          #len(lon)=1440
tempN = Datifiltrati.variables['t2m'][:] '''        #len(temp)=365

lat_range = list(range(40,len(lat)))
lon_range = list(range(0,len(lon)))


for y in lat_range:
    for x in lon_range:
        print(y, '          ', x )
        timeseries = temp[:,y,x]
        timeseriesDM = temp[:ndays,y,x]
        n=0
        b=0
        for yr in Nyear:                     
            if yr in normalyear:
                for d in Nday:
                    #print(d)
                    dailymean1[d] += timeseriesDM[365*n+366*b+d]/nyear
                n += 1
            if yr in biyear:
                for d in Nbiday:
                    if d < 59:
                        dailymean1[d] += timeseriesDM[365*n+366*b+d]/nyear
                    if d == 59:
                        dailymean1[365] += timeseriesDM[365*n+366*b+59]/nbiyear             
                    if d > 59:
                        dailymean1[d-1] += timeseriesDM[365*n+366*b+d]/nyear
                b+=1   

        n=0
        b=0
        #                               DATA FILTERED
        for yr in Totyear:
            if yr in normalyear:
                for d in Nday:
                    temp_nuova[365*n+366*b+d,y,x] = timeseries[365*n+366*b+d] - dailymean1[d]
                n += 1
            if yr in biyear:
                for d in Nbiday:
                    if d < 59:
                        temp_nuova[365*n+366*b+d,y,x]= timeseries[365*n+366*b+d] - dailymean1[d]
                    if d == 59:
                        temp_nuova[365*n+366*b+d,y,x]= timeseries[365*n+366*b+d] - dailymean1[365]               
                    if d > 59:
                        temp_nuova[365*n+366*b+d,y,x]= timeseries[365*n+366*b+d] - dailymean1[d-1]
                b+=1   
        
        '''plt.scatter(timerange, timeseries, label='timeseries1')
        plt.scatter(list(range(0,366)), dailymean1, label='dailmean')
        plt.scatter(timerange, temp_nuova[:,y,x], label='tempnuova')
        plt.xlabel('Day')
        plt.ylabel('Temp')
        plt.legend()
        plt.show()'''
        dailymean1 = [0]*366








'''# Scrivi i dati modificati nella variabile 'temp' nel nuovo file
lat2 = Datifiltrati.variables['lat'][:]          #len(lat)=721
lon2 = Datifiltrati.variables['lon'][:]          #len(lon)=1440
temp2 = Datifiltrati.variables['t2m'][:]         #len(temp)=365

print(lat2)
print(lon2)
print(temp2)'''

# Chiudi i file per salvare le modifiche
data.close()
Datifiltrati.close()