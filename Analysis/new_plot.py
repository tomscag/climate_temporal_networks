### New Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import shiftgrid

import networkx as nx

def generate_coordinates(sizegrid):
    lats = np.arange(-90,90+sizegrid,sizegrid,dtype=float)  # 37 
    lons = np.arange(-180,180,sizegrid,dtype=float)         # 72
    N = len(lons)*len(lats)
    coords = {key:None for key in range(N)}
    node = 0
    for lat in lats:
        for lon in lons:
            coords[node] = [lat,lon]
            node += 1
    return coords, lons, lats


def plot_earth(data_matrix,sizegrid=5):
    ''' Plotting earth '''
    
    # m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
    m = Basemap(projection='robin', lat_0=0, lon_0=0) # If you shift you have to shift lon lat too
    fig, ax = plt.subplots(figsize=(12, 9))
    lats = np.arange(-90,90+5,5,dtype=float)  # 37 
    lons = np.arange(-180,180,5,dtype=float)         # 72

    lon2, lat2 = np.meshgrid(lons, lats)
    # data_matrix = np.random.rand(len(lats),len(lons))
    x, y = m(lon2,lat2) # Convert to meters
    cs = m.contourf(x,y,data_matrix)

    
    m.drawcoastlines()
    m.fillcontinents(color='gray',lake_color='gray',alpha=0.45)
    m.drawmapboundary()
    m.drawcountries()
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True], linewidth=0.5, color='grey')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False], linewidth=0.5, color='grey')
    # m.colorbar(location='right', label='Degree')

# plot_earth()



def create_fuzzy_network(edgelist):
    rnd = np.random.rand(*edgelist['prob'].shape)
    edgelist = edgelist.loc[ rnd < edgelist['prob'] ]
    

    G = nx.from_pandas_edgelist(edgelist,source="node1",target="node2")

    return G


def total_degree_nodes(G):

    # lats = np.arange(-90,90+5,5,dtype=float)  # 37 
    # lons = np.arange(-180,180,5,dtype=float)         # 72

    
    coords, lons, lats = generate_coordinates(sizegrid=5)
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

###################################
###################################

folderpath = "./Output_cluster/"
fnameinput = "year_1976_maxlag_150.csv"

name = folderpath + fnameinput

df = pd.read_csv(name,sep="\t",header=None,names=["node1","node2","zscore","maxlag","prob"])










