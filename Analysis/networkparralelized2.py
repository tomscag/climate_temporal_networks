from netCDF4 import Dataset
import matplotlib.pyplot as plt
import math
import random
from math import e 
import pandas as pd
import numpy as np
import statistics
import iaaft
import random
import os
import time
import multiprocessing as mp


import cProfile
import pstats
from numpy.lib.stride_tricks import as_strided

def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def crosscorrelation(x, y, maxlag, normalize=True):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.

    https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy
    """

    if normalize:
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) /  np.std(y)        

    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)





def crosscorr(datax, datay, lag):
    return datax.corr(datay.shift(lag), method='pearson')

def zscore_lag(timeserie1, timeserie2, num_surr, max_lag = 180):
    """
    INPUT
        num_surr:   number of surrogates
        max_lag:    maximum lag to compute cross-correlation
    """

    serie1 = pd.Series(timeserie1) 
    serie2 = pd.Series(timeserie2)

    cross_corr = crosscorrelation(serie1, serie2, max_lag)
    crossmax   = cross_corr.max()
    the_lagmax = cross_corr.argmax() - (max_lag + 1)
    
    sh_serieiaaft1 = iaaft.surrogates(x=timeserie1, ns= num_surr, verbose=False)
    sh_serieiaaft2 = iaaft.surrogates(x=timeserie2, ns= num_surr, verbose=False)

    cross_corr_surr = []

    for n in range(0,num_surr):
        serie1_surr = pd.Series(sh_serieiaaft1[n])
        serie2_surr = pd.Series(sh_serieiaaft2[n])
        sh_cross_corr = crosscorrelation(serie1_surr, serie2_surr, max_lag)
        sh_crossmax   = sh_cross_corr.max()
        cross_corr_surr.append(sh_crossmax)
    
    crosscorr_surr_mean  = statistics.mean(cross_corr_surr)
    crosscorr_surr_stdev = statistics.pstdev(cross_corr_surr)        
    Z_valuemax = abs(crossmax - crosscorr_surr_mean)/crosscorr_surr_stdev

    return Z_valuemax, the_lagmax #, Z_valuemin, the_lagmin


def pval_erfc(zeta):
    return (1-math.erf(zeta/(2**0.5)))

def pval_cheb(zeta):
    if zeta < 1:
        pval =1
    else:
        pval = 1/(zeta**2)

    return pval

def probabilityfrom(Z, dis):
    if Z < 1:
        pval =1
    else:
        pval = 1/(Z**2)
    
    if pval < e**(-1):
        B_value = -e*pval*math.log(abs(pval))
    else:
        B_value = 1
    
    prior = math.exp(-dis/2000)
    prob = 1-(1+((B_value)*(1-prior)/(prior))**(-1))**(-1)
    print(f"Prob: {prob:.3f} prior: {prior:.3f} dist:{dis} Bval: {B_value}")

    return prob

def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371 #avarege radius

    # degree to radiant
    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180

    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    # haversine formula
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance



def analyze(indi, nodes):
    Ai,Aj = nodes[indi]
    for indj,node in enumerate(nodes):
        (Bi, Bj) = nodes[indj]
        
        if indi < indj: # Undirected network
            print(f"Analyzing point of indices: ({Ai} {Aj}) ({Bi} {Bj}) ")
            ddd = haversine_distance(lat[Ai], lon[Aj], lat[Bi], lon[Bj])

            for dec in range(periods):
    
                Zsmx, lagmx  = zscore_lag(temp[start_days[dec]:start_days[dec+1],Ai,Aj], temp[start_days[dec]:start_days[dec+1], Bi, Bj], numiaaft, max_lag)
                prob = probabilityfrom(Zsmx, ddd)

                nome_file = f"{foutpath}/network_period{dec}.txt"
                with open(nome_file, "a+") as file:
                    file.write(f"{indi}\t{indj}\t{prob}\t({lat[Ai]},{lon[Aj]})\t({lat[Bi]},{lon[Bj]})\t{lagmx}\n")

                nome_file2 = f"{foutpath2}/violin_period{dec}.txt"
                with open(nome_file2, "a+") as file:
                    file.write(f"{ddd} \t {Zsmx} \n") 







############################################
############################################
############################################


# DATA INPUT
data = Dataset('./data/t2m/filtered_t2m_1970_2022_4grid.nc', 'r')
lat  = data.variables['lat']        
lon  = data.variables['lon']            
temp = data.variables['t2m']

# DATA OUTPUT
foutpath = "./Analysis/Output"
if not os.path.exists(foutpath):
    os.makedirs(foutpath)

foutpath2 = "./Analysis/Output/Violinplot"
if not os.path.exists(foutpath2):
    os.makedirs(foutpath2)
        

# PARAMETERS
years_analysed = 5      #five years intervals
tot_years = 2022-1970
periods = tot_years//years_analysed
start_days = []
for dec in range(0,(periods)+1):
    start_days.append(365*years_analysed*(dec)+((1+dec*years_analysed)//4))     #for leap years
max_lag = 30*6           #six months
num_edgelists = 50
numiaaft = 30

# NODES
'''
lon_range = range(0,len(lon), 2)
lat_range = range(0,len(lat), 2)
nodes = tuple((i,j) for i in lat_range for j in lon_range)'''

nodes = [(2, 0), (2, 30), (2, 60), (7, 0), (20,45)]

if __name__ == "__main__":

    pool = mp.Pool(7)   # Use the number of cores of your PC
    for indi,nod in enumerate(nodes):
        
        pool.apply_async(analyze, args = (indi, nodes, )) # Parallelize
        #analyze(indi,nodes)   # Uncomment to not parallelize
    pool.close()
    pool.join()
    data.close()


# if __name__ == "__main__":
#     with cProfile.Profile() as profile:
#         pool = mp.Pool(7)   # Use the number of cores of your PC
#         for indi,nod in enumerate(nodes):
            
#             # pool.apply_async(analyze, args = (indi, nodes, )) # Parallelize
#             analyze(indi,nodes)   # Uncomment to not parallelize
#         pool.close()
#         pool.join()
#         data.close()
#         results = pstats.Stats(profile)
#         results.sort_stats(pstats.SortKey.TIME)
#         results.print_stats()
#         results.dump_stats("results4.profile")


