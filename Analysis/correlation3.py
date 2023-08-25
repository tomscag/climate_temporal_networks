from netCDF4 import Dataset
import matplotlib.pyplot as plt
import math
import random
from math import e 
import pandas as pd
import numpy as np
import statistics

def crosscorr(datax, datay, lag):
    """Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag), method='spearman')


from tqdm import tqdm

def surrogates(x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"):
    """
    Returns iAAFT surrogates of given time series.

    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.

    See Also
    --------
    numpy.argsort

    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs


################ DATA ##############


data = Dataset('./data/filtered_t2m_1970_2022_2grid.nc', 'r')
lat = data.variables['lat'][:]         
lon = data.variables['lon']            
temp = data.variables['t2m'][:]        


###############  PARAMETERS #########

nyear    = 1
N_simu   = 100         
lagRange = list(range(0, 365))
lagRange = list(range(0,50))
Ai = 45                #latitudine
Aj = 90                 #longitudine
Bi = 45
Bj = 115

#   INIZIALIZATION
nbiyear = (nyear+1) // 4 
nnormyear = nyear - nbiyear
ndays = nbiyear * 366 + nnormyear *365
lagsP = []
sh_cross_list = []
sh_lagsP = []
cross_corrsP = []
sh_cross_corrsP = []
crossmax = 0.0001
sh_crossmax = 0.00001

#   ORIGINAL TIMESERIES
timeseries1 = temp[-ndays:,Ai,Aj]
timeseries2 = temp[-ndays:,Bi,Bj]
serie1 = pd.Series(timeseries1) 
serie2 = pd.Series(timeseries2)

for lag in lagRange:   #\s
    cross1 = crosscorr(serie1, serie2, lag)
    #   GRAPH
    lagsP.append(lag)               
    cross_corrsP.append(cross1)
    #   FIND THE LAG WHERE I HAVE MAX CORRELATION
    if abs(cross1) > crossmax:        
        crossmax = cross1
        the_lag = lag

sh_serie1_list = surrogates(timeseries1,N_simu)
sh_serie2_list = surrogates(timeseries2,N_simu)
for n in range(0,N_simu-1):
    sh_serie1 = pd.Series(sh_serie1_list[n,:])
    sh_serie2 = pd.Series(sh_serie2_list[n,:])
    sh_cross1 = crosscorr(sh_serie1, sh_serie2, the_lag)        #LA CORRELAZIONE LA CALCOLO SOLO AL LAG 
    sh_cross_list.append(sh_cross1) # Cross correlation
    sh_lagsP.append(the_lag)


sh_crossmax = max(sh_cross_list)


sh_cross_mean = statistics.mean(sh_cross_list)
sh_cross_stdev = statistics.pstdev(sh_cross_list) 
Z_value = abs(crossmax - sh_cross_mean)/sh_cross_stdev            #CHOOSE BETWEEN SH_CROSSMAX OR SH_CROSS_MEAN
p_value = 1-math.erf(Z_value/(2**0.5))

print("The Lag:                              ", the_lag)
print("media delle shuffled:                 ", sh_cross_mean)
print("max delle shuffled:                   ", sh_crossmax)
print("stdev delle shuffled:                 ", sh_cross_stdev)
print("correlazione massima TS originale:    ", crossmax)
print("Z-value:                              ", Z_value)
print("pval: 1 - erf(Z/sqrt(2)):             ", p_value) 
print("probability                           ", 1- p_value) 

# PLOT
plt.scatter(sh_lagsP, sh_cross_list, label='Shuffled')
plt.scatter(lagsP, cross_corrsP, label='Original')
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.legend()
plt.show()




