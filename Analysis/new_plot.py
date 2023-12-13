### New Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import shiftgrid

import networkx as nx

from lib.misc import generate_coordinates
from lib.plot import  PlotterEarth



######################################
######################################
######################################

def create_fuzzy_network(edgelist):
    ''''
        Generate fuzzy network from edgelist
    '''
    
    rnd = np.random.rand(*edgelist['prob'].shape)
    edgelist = edgelist.loc[ rnd < edgelist['prob'] ]
    

    G = nx.from_pandas_edgelist(edgelist,source="node1",target="node2",edge_attr=True)
    G.add_nodes_from(range(2664)) # In case some nodes are missing in the edgelist

    return G


def total_degree_nodes(G):
    """

    """

    
    coords, lons, lats = generate_coordinates(sizegrid=5)
    # Only degree
    data_matrix1 = np.array(list(dict(sorted(G.degree())).values())).reshape(len(lats),len(lons))

    W = {key:None for key in sorted(G.nodes())} # Weighted connectivity
    for node in G.nodes():

        w=0
        for item in G[node]:
            w += np.abs(np.cos(coords[item][0]*2*np.pi/360)) # cos lat
        W[node] = w

    c = 0
    for key,value in coords.items():
        c += np.abs(np.cos(value[0]*2*np.pi/360))

    W = {key:value/c  for key,value in W.items()}
    data_matrix = np.array(list(W.values())).reshape(len(lats),len(lons))
    return data_matrix,data_matrix1

def load_data(name,lag_bounds = [0,10]):
    df = pd.read_csv(name,sep="\t",header=None,
                     names=["node1","node2","zscore","maxlag","prob"]
                     )
    df = df.where( (df["maxlag"] >= lag_bounds[0]) & (df["maxlag"] <= lag_bounds[1]) ).dropna()
    # df['prob'].loc[ (df['maxlag'] < lag_bounds[0]) & df['maxlag'] > lag_bounds[1] ] = 0
    df[['node1','node2']] = df[['node1','node2']].astype(int)
    return df

###################################
###################################

year       = 1990
plev       = 750    # Pressure level
folderpath = f"./Output/correlations/plev_{plev}/"
fnameinput = f"temperature_press_{plev}_year_{year}_maxlag_150.csv"
lag_bounds = [-10,20]
name = folderpath + fnameinput
resfolder  = "./fig/"
proj = "robin"      # Earth projection
savefig = True

df = load_data(name,lag_bounds)


graph = create_fuzzy_network(df)
data_matrix,data_matrix1 = total_degree_nodes(graph)

# plot_earth(data_matrix,sizegrid,proj,savefig)

# plt.title(f"{year}  lags: {str(lag_bounds[0])}-{str(lag_bounds[1])}",fontsize=18)



plote = PlotterEarth(proj,year,resfolder)

## Plot linemap
node = 1050
plote.plot_linemap(graph,node)

print("ciao")
## Plot heatmap
# plote.plot_heatmap(data_matrix)