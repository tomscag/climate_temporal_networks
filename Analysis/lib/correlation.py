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
def cross_correlation(
        x: np.ndarray, 
        y: np.ndarray, 
        maxlag: int, 
        normalize: bool = True
        ) -> np.ndarray:
    """
    Compute the cross-correlation between two 1D arrays with a limited lag window.

    This function calculates the cross-correlation of two equally sized
    one-dimensional arrays, `x` and `y`, restricting the lag range to
    `[-maxlag, +maxlag]`. The implementation is equivalent to taking a slice
    of `numpy.correlate(x, y, mode='full')`, but it is more efficient when only
    a limited number of lags are required.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array.
    y : np.ndarray
        Input 1D array of the same length as `x`.
    maxlag : int
        Maximum lag (in number of samples) for which to compute the correlation.
        The output covers lags from `-maxlag` to `+maxlag`.

    Returns
    -------
    np.ndarray
        Cross-correlation values for lags in the range `[-maxlag, +maxlag]`.
        The output array has length `2 * maxlag + 1`.

    Notes
    -----
    - The cross-correlation is defined here as:
        R_xy(τ) = Σ_t x[t] * y[t + τ]
      where τ ranges from `-maxlag` to `+maxlag`.

    - This function uses stride tricks for efficiency. It is suitable for
      just-in-time (JIT) compilation with Numba, since `np.pad` is avoided.

    References
    ----------
    Stack Overflow discussion:
        https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy

    Examples
    --------
        rng = np.random.default_rng(42)
        T = 1000
        tau = 3
        maxlag = 10
        
        x = rng.normal(size=(T,))
        y = rng.normal(size=(T,))
        
        for i in range(T-tau):
            y[i+tau] = 0.2*x[i] + 0.1*rng.normal()
            
        c = cross_correlation(y, x, maxlag = maxlag)
    """

    # if normalize==True: # FOR SPEED: the series are already passed normalized and rescaled
    #     x = (x - np.mean(x)) / (np.std(x) * len(x))
    #     y = (y - np.mean(y)) /  np.std(y)  

    # x = _check_arg(x, 'x')
    # y = _check_arg(y, 'y')
    # py = np.pad(y.conj(), 2*maxlag, mode='constant')
    
    # Pad y with zeros on both sides (Numba-compatible)
    py = np.concatenate( (np.zeros(2*maxlag), y.conj(), np.zeros(2*maxlag))) 

    # Create a strided view of y for efficient lagged multiplication
    T = as_strided(
        py[2*maxlag:], 
        shape=(2*maxlag+1, len(y) + 2*maxlag),
        strides=(-py.strides[0], py.strides[0])
        )
    # px = np.pad(x, maxlag, mode='constant')
    px = np.concatenate( (np.zeros(maxlag), x.conj(), np.zeros(maxlag)))

    # Compute the dot product for each lag    
    return T.dot(px)