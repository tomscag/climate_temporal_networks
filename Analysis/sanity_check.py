import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

from Functions.correlations_climate import _check_arg, crosscorrelation
from Functions import iaaft
import statistics
import math

from netCDF4 import Dataset


#############################
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


def import_dataset(size):
    '''
    OUTPUT
        data: (2d-array)
            data reshaped

        indices: (1d-array)
            indices 1 Jan

        nodes: (dict)
            label: (lat,lon)
    '''


    data = Dataset(f'../data/t2m/anomalies_t2m_1970_2022_{size}grid.nc', 'r')
    indices = first_day_of_year_index(data)
    lat  = data.variables['lat']        
    lon  = data.variables['lon']            
    temp = data.variables['t2m']
    data = np.array(temp).reshape( temp.shape[0],temp.shape[1]*temp.shape[2]) # time, lat * lon
    
    count = 0
    nodes = dict()
    for item_lat in enumerate(lat):
        for item_lon in enumerate(lon):
            nodes[count] = (float(item_lat[1].data),float(item_lon[1].data))
            count += 1
            
    return data, indices, nodes


def first_day_of_year_index(data):
    '''
        Return the indices of the first day of the years
    '''
    doy = np.array(data['dayofyear']) 
    return np.where( doy == 1)[0]


def plot(vec):
    plt.plot(vec)
    plt.savefig('test.png')




def posterior_link_probability_iaaft(x,y,crossmax,dist,num_surr=50):

    surr_x = iaaft.surrogates(x=x, ns= num_surr, verbose=False)
    surr_y = iaaft.surrogates(x=y, ns= num_surr, verbose=False)

    cross_corr_surr = []

    for n in range(0,num_surr):
        serie1_surr = pd.Series(surr_x[n])
        serie2_surr = pd.Series(surr_y[n])
        sh_cross_corr = crosscorrelation(serie1_surr, serie2_surr, max_lag)
        sh_crossmax   = sh_cross_corr.max() 
        cross_corr_surr.append(sh_crossmax)
    
    crosscorr_surr_mean  = statistics.mean(cross_corr_surr)
    crosscorr_surr_stdev = statistics.pstdev(cross_corr_surr)        
    Z = abs(crossmax - crosscorr_surr_mean)/crosscorr_surr_stdev


    if Z < 1:
        pval =1
    else:
        pval = 1/(Z**2)
    
    if pval < math.e**(-1):
        B_value = -math.e*pval*math.log(abs(pval))
    else:
        B_value = 1
    
    # Prior probaility for the null hypothesis
    K = 2000
    prior = 1 - math.exp(-dist/K)

    # Posterior probability of link existence
    prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)

    return prob


def posterior_link_probability_havlin(cross_corr,dist,max_lag):
    '''
        We compute the zscore as explained in "Stability of Climate Networks with Time"
        https://doi.org/10.1038/srep00666
    '''
    crossmax   = max(abs(cross_corr))
    the_lagmax = abs(cross_corr).argmax() - (max_lag + 1)

    Z = (max(abs(cross_corr)) - cross_corr.mean() )/ cross_corr.std()


    if Z < 1:
        pval =1
    else:
        pval = 1/(Z**2)
    
    if pval < math.e**(-1):
        B_value = -math.e*pval*math.log(abs(pval))
    else:
        B_value = 1
    
    # Prior probaility for the null hypothesis
    K = 2000
    prior = 1 - math.exp(-dist/K)

    # Posterior probability of link existence
    prob = 1-(1+((B_value)*(prior)/(1-prior))**(-1))**(-1)

    return Z, prob,crossmax,the_lagmax

def save_results(i,j,Z,the_lagmax,prob,foutput):
    with open(foutput,'a') as file:
        file.write(f"{i}\t{j}\t{Z:.4f}\t{the_lagmax}\t{prob:.4f}\n")
    


def correlation_all(data,foutput):
    T,N = data.shape

    for i in range(N):
        print(f"Computing node {i}")
        for j in range(i+1,N):
            dist = haversine_distance( nodes[i][0],nodes[i][1], nodes[j][0],nodes[j][1])
            x  = data[:,i]
            y  = data[:,j]
            cross_corr = crosscorrelation(x, y, max_lag)
            # crossmax   = cross_corr.max()   # Put abs(cross_corr) to consider negative lags too
            # the_lagmax = cross_corr.argmax() - (max_lag + 1)

            # prob = posterior_link_probability_iaaft(x,y,crossmax,dist,num_surr=50)
            Z, prob,crossmax,the_lagmax = posterior_link_probability_havlin(cross_corr,dist,max_lag)
            if prob > 1e-2:
                save_results(i,j,Z,the_lagmax,prob,foutput)

            # print(f"Computing nodes {i} and {j}: corrmax {crossmax:.4f} at lag {the_lagmax}, prob {prob:.4f}")


#############################

# Parameters

size = 5    # Size of the grid in degree


# Load data
data, indices, nodes = import_dataset(size)





max_lag = 150
years   = range(1970,2022+1)  # from 1970 to 2022
year    = 3
foutput = f'year_{years[year]}_maxlag_{max_lag}.csv'

correlation_all(data[indices[year]:indices[year+1],:],foutput)



# 
# x = data[indices[year]:indices[year+1],100]
# y = data[indices[year]:indices[year+1],214]
# crosscorrelation(x, y, max_lag)