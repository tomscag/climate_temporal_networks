# Analysis


import numpy as np
import pandas as pd
import networkx as nx
import random
import time 
import matplotlib.pyplot as plt
import math

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_networks(G, num_networks):
    ''' 
    INPUT
        G:
            edge list representing the probability that an edge exists

        num_networks:
            number of fuzzy networks to generate
    '''
    G_list = [nx.Graph() for n in range(num_networks)]
    for net in range(num_networks):
        for a,b in G.edges():
            if random.random() < G.get_edge_data(a,b)['prob']:
                G_list[net].add_edge(a,b)

    return G_list

def analyze_networks(G_list):
    pass

def extract_coordinates():
    nodes_coord = {}
    for nodeA, nodeB in G.edges():
        latA, lonA = eval(G[nodeA][nodeB]['coord_A']) 
        latB, lonB = eval(G[nodeA][nodeB]['coord_B']) 
        nodes_coord[nodeA] = (latA, lonA)
        nodes_coord[nodeB] = (latB, lonB)
    lat = [value[0] for key,value in nodes_coord.items() ]
    lon = [value[1] for key,value in nodes_coord.items() ]

    return nodes_coord, lat,lon

def compute_area_weighted_connectivity(nodelist):

    N = sum( [ math.cos( math.radians(nodes_coord[item][0])) for item in nodelist ] )
    
    return N/D
    

def plot_earth():
    ''' Plotting earth '''
    fig, ax = plt.subplots(figsize=(12, 9))
    # m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
    m = Basemap(projection='robin', lat_0=0, lon_0=-90)
    m.drawcoastlines()
    m.fillcontinents(color='gray',lake_color='gray')
    m.drawmapboundary()
    m.drawcountries()
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True], linewidth=0.5, color='grey')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False], linewidth=0.5, color='grey')

    lon_list = sorted(set(nodes['lon']))
    lat_list = sorted(set(nodes['lat']))

    lon2, lat2 = np.meshgrid(lon_list,lat_list) 
    x, y = m(lon2, lat2)


    data_matrix = np.random.randn(len(lon_list),len(lat_list))
    for i in range(0,len(lon_list)):
        for j in range(0,len(lat_list)):
            data_matrix[i,j] = 0

    for index, row in nodes.iterrows():
        i = lon_list.index(row['lon'])
        j = lat_list.index(row['lat'])
        data_matrix[i,j] = compute_area_weighted_connectivity(G_list[0][index])

    cs = m.contourf(x,y,data_matrix.T,200, cmap=plt.cm.rainbow)
    m.colorbar(location='right', label='Area weighted connectivity')

    return ax



# float(G[1][2]['coord_A'].split(",")[0].split("(")[1]) # Extract coord

# [val for _,val in G_list[0].degree()] # degree

############################
############################
############################

st = time.time()


num_networks = 100


# HEATMAP

max_degree = 0.1
min_degree = 0
cmap = plt.cm.Reds
norm = plt.Normalize(min_degree, max_degree)

_fpath = "./Output/276_nodes/network_period0.txt"
nodes  = pd.read_csv("./Output/276_nodes/network.nodelist",sep=" ",index_col=0,names=["label","lon","lat"])

G = nx.read_edgelist(_fpath, delimiter="\t",nodetype=int, data=(("prob", float),("coord_A",str),("coord_B",str),("lag",int)))


G_list = generate_networks(G, num_networks)

nodes_coord,lat,lon = extract_coordinates()

# with open("./network.nodeslit","w") as file:
#     for items in nodes_coord.items():
#         file.write(f"{items[0]} {items[1][1]} {items[1][0]}\n")


D = sum( [ math.cos( math.radians(nodes_coord[item][0])) for item in G.nodes()] )  # Normalization weighted connectivity


ax = plot_earth()

et = time.time()



print(f"{et-st:.3f} seconds")
print("finished")
plt.show()