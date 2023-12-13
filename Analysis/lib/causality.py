import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests





def test_granger():
    N = 1000
    x = np.random.rand(N)
    y = np.zeros(N)
    y[1:] = x[0:N-1] + 0.2*np.random.random(N-1)
    data = np.array([x,y]).T
    res = grangercausalitytests(np.array([x,y]).T,maxlag=2)
    print(res)

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):  
    import pandas as pd  
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    maxlag = 1
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    return df
