### New Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import shiftgrid

import networkx as nx

from lib.plot import  PlotterEarth
from lib.misc import (
            generate_coordinates,
            create_fuzzy_network, 
            create_full_network,
            load_edgelist,
            filter_network_by_distance
            )


######################################
######################################
######################################


def total_degree_nodes(G):
    """
        Return:
            weights_matrix
                Each entry is a node, organized by lat x lon,
                and represents the total area to which a node
                is linked to
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
    weights_matrix = np.array(list(W.values())).reshape(len(lats),len(lons))
    return weights_matrix,data_matrix1


def plot_teleconnections(plote,elist,node=[1050], K = 2000):

    # Filter out short links
    elist = filter_network_by_distance(elist,K )

    # Create the full network "weighted" with the edge-probabilities
    graph = create_full_network(elist)

    ## Plot linemap
    plote.plot_linemap(graph,node)



###################################
###################################


def main():

    year       = 1990
    plev       = 750    # Pressure level
    folderpath = f"./Output/correlations/plev_{plev}/"
    fnameinput = f"temperature_press_{plev}_year_{year}_maxlag_150.csv"
    lag_bounds = [-10,20]
    name = folderpath + fnameinput
    resfolder  = "./fig/"
    proj = "robin"      # Earth projection
    savefig = True

    elist = load_edgelist(name,lag_bounds)

    # Filter out short links
    # elist = filter_network_by_distance(elist, K = 2000)

    # Create a network sample
    graph = create_fuzzy_network(elist)


    weights_matrix,data_matrix1 = total_degree_nodes(graph)

    # plot_earth(weights_matrix,sizegrid,proj,savefig)

    # plt.title(f"{year}  lags: {str(lag_bounds[0])}-{str(lag_bounds[1])}",fontsize=18)



    plote = PlotterEarth(proj,year,resfolder)

    plot_teleconnections(plote,elist,node=list(range(2664)),K=5000)
    

    ## Plot heatmap
    # plote.plot_heatmap(weights_matrix)


if __name__ == "__main__":
    main()