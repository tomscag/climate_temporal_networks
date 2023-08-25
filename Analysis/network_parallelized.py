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

def zscore_lag(timeserie1, timeserie2, num_surr, max_lag = 366):
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

        cross_corr_surr.append( serie1_surr.corr(serie2_surr, method='pearson') )
    
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
    print(f"BValue: {B_value:.3f} prior: {prior:.3f}")
    prob = 1-(1+((B_value)*(1-prior)/(prior))**(-1))**(-1)

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

            for dec in list(range(0,5)):
                # print(Ai, '\t', Aj, '\t',Bi, '\t', Bj, '\t', dec)
    
                Zsmx, lagmx  = zscore_lag(temp[start_days[dec]:start_days[dec+1],Ai,Aj], temp[start_days[dec]:start_days[dec+1], Bi, Bj], numiaaft)

                prob = probabilityfrom(Zsmx, ddd)

                nome_file = f"{foutpath}/network_decade{dec}.txt"
                with open(nome_file, "a+") as file:
                    file.write(f"{indi} \t {indj} \t {prob} \t ({Ai} {Aj}) \t ({Bi} {Bj}) \n") 





############################################
############################################
############################################


# DATA INPUT
data = Dataset('./data/t2m/filtered_t2m_1970_2022_4grid.nc', 'r')
lat  = data.variables['lat']        
lon  = data.variables['lon']            
temp = data.variables['t2m']

foutpath = "./Analysis/Output"

if not os.path.exists(foutpath):
    os.makedirs(foutpath)

        

# PARAMETERS
numiaaft = 30
start_days = (0,3652,7305,10957,14610,18262)    #ten years intervals
lon_range = list(range(0, 3))
lat_range = list(range(0, len(lat)))
num_edgelists = 50
count = 0

# NODES
nodes = []
for i in range(5,45,10):
    for j in range(5,90,10):
        nodes.append((i,j))

# nodes = tuple((i,j) for i in range(5,45,10) for j in range(5,90,10))
nodes = [(2, 0), (2, 30), (2, 60), (7, 0)]

if __name__ == "__main__":

    pool = mp.Pool(7)   # Use the number of cores of your PC
    for indi,nod in enumerate(nodes):
        
        # pool.apply_async(analyze, args = (indi, nodes, )) # Parallelize
        analyze(indi,nodes)   # Uncomment to not parallelize
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


