# Test correlations


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing as mp
from timeit import default_timer as timer
from numpy.lib.stride_tricks import as_strided

###########################################


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

    # if normalize:
    #     x = (x - np.mean(x)) / (np.std(x)*len(x) )
    #     y = (y - np.mean(y)) /  (np.std(y) )        

    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')

    
    crossc = T.dot(px)

    if normalize:
        for lag in range(-maxlag,maxlag+1):
            crossc[lag+maxlag] = crossc[lag+maxlag]/( len(x)-lag )

    return crossc



def cross_correlation_from_definition(x,y,maxlag,normalize):

    crossc = np.zeros(2*maxlag+1)
    c = 0
    for lag in range(-maxlag,maxlag+1):
        s = 0
        if lag>= 0:
            for t in range(N-lag):
                s += x[t+lag]*y[t]
            crossc[c] = s
            c += 1
        elif lag <0:
            for t in range(np.abs(lag),N):
                s += x[t+lag]*y[t]
            crossc[c] = s
            c += 1
    
    if normalize:
        for lag in range(-maxlag,maxlag+1):
            crossc[lag+maxlag] = crossc[lag+maxlag]/( len(x)-lag )

    return crossc



def cross_correlation_numpy(x,y,maxlag,normalize):

    N = len(x)
    res = np.correlate(x, y, mode='full')[N-maxlag-1:N+maxlag]

    if normalize:
        res = res/N

    return res

#########################################
#########################################
#########################################

N = 200000

# Generate two correlated random series
x = np.random.rand(N)*10
y = 0.3*x + np.random.rand(N)*10
# y = x

maxlag = 10
normalize = True

# Definition
start2 = timer()
res2 = cross_correlation_from_definition(x,y,maxlag,normalize)
end2   = timer()

# Numpy function cross-correlation
start = timer()
res = cross_correlation_numpy(x,y,maxlag,normalize)
end = timer()


# Our new function
start1 = timer()
res1 = crosscorrelation(x, y, maxlag, normalize=normalize)
end1 = timer()

# Print
lag_test = 3
print(f"Definition:\t{res2[lag_test]}\t time: {end2-start2} sec")
print(f"Numpy:\t\t{res[lag_test]}\t time: {end-start} sec")
print(f"Our function:\t{res1[lag_test]}\t time: {end1-start1} sec")


