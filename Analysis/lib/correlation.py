# List of functions
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import jit

# @jit(nopython=True)
def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

@jit(nopython=True)
def cross_correlation(x, y, maxlag, normalize=True):

    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.

    https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy
    """

    # if normalize: # No need to normalize if series are already normalized 
    #     x = (x - np.mean(x)) / (np.std(x) * len(x))
    #     y = (y - np.mean(y)) /  np.std(y)        

    # x = _check_arg(x, 'x')
    # y = _check_arg(y, 'y')
    # py = np.pad(y.conj(), 2*maxlag, mode='constant')
    py = np.concatenate( (np.zeros(2*maxlag), y.conj(), np.zeros(2*maxlag))) # equivalent of pad (not supported in numba)

    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    # px = np.pad(x, maxlag, mode='constant')
    px = np.concatenate( (np.zeros(maxlag), x.conj(), np.zeros(maxlag)))    
    return T.dot(px)