# Analisi prior

import pandas as pd
from Functions.other_functions import generate_coordinates, haversine_distance


import multiprocessing as mp


def analyze_zscore(fileinput):
    df = pd.read_csv(fileinput,sep="\t",header=None,names=["node1","node2","zscore","maxlag","prob"])
    # df = df.where( (df["maxlag"] >= lag_bounds[0]) & (df["maxlag"] <= lag_bounds[1]) ).dropna()
    # df['prob'].loc[ (df['maxlag'] < lag_bounds[0]) & df['maxlag'] > lag_bounds[1] ] = 0

    df['dist'] = df.apply(
                lambda x: haversine_distance(
                            coords[x.node1][0], coords[x.node1][1],
                            coords[x.node2][0], coords[x.node2][1]), 
                        axis=1)
    return df

######################################



if __name__ == "__main__":

    # Parameters
    variable = 't2m'   # folders in ../data/
    size = 5    # Size of the grid in degree

    coords, lons, lats = generate_coordinates(size)

    # Load results
    fileinput = f'./Output_cluster/analisi_5grid/year_1970_maxlag_150.csv'

    years   = range(1970,2022)  # from 1970 to 2022


    # pool = mp.Pool(8)   # Use the number of cores of your PC

    for year,y in enumerate(years):
        foutput = f'./Output/GC/year_{years[year]}_maxlag_{max_lag}.csv'    
        # pool.apply_async(granger_causality_all, args = (data[indices[year]:indices[year+1],:], foutput, )) # Parallelize
        granger_causality_all(data[indices[year]:indices[year+1],:],foutput)  # Uncomment to not parallelize
    # pool.close()    
    # pool.join()