# Analisi prior

import pandas as pd
import matplotlib.pyplot as plt
from Functions.other_functions import generate_coordinates, haversine_distance




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


def plot_distance(df):
    plt.scatter(df['dist'],df['zscore'],alpha=0.1)


######################################



if __name__ == "__main__":

    # Parameters
    variable = 't2m'   # folders in ../data/
    size = 5    # Size of the grid in degree

    coords, lons, lats = generate_coordinates(size)

    # Load results file
    fileinput = f'./Output/correlations/plev_750/temperature_press_750_year_1970_maxlag_150.csv'

    df = analyze_zscore(fileinput)
    plot_distance(df)