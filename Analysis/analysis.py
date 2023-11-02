import random
import networkx as nx
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#PARAMETERS
periods = 1
n_graphs = 50


# COORDINATES
nodes_coordinates = {}
with open(f'./Output/network_period0.txt', 'r') as f:
    for line in f.readlines():
        parts = line.split("\t")      
        nodeA = parts[0]
        latA, lonA = eval(parts[3]) 
        nodes_coordinates[nodeA] = (latA, lonA)


for p in range(periods):
    edges = []

    with open(f'./Output/network_period{p}.txt', 'r') as f:
        for line in f.readlines():
            nodoA, nodoB, probability, *_ = line.split()
            edges.append((nodoA, nodoB, float(probability)))           

    # GRAPHS GENERATION
    graphs = []
    for _ in range(n_graphs):
        edge_list = []
        for nodoA, nodoB, probability in edges:
            if random.random() < probability:
                edge_list.append((nodoA, nodoB))
        graphs.append(edge_list)



    total_degrees = defaultdict(int)        #dictionary for degrees
    for edge_list in graphs:
        G = nx.Graph()  
        G.add_edges_from(edge_list)  

        for node, degree in G.degree():
            total_degrees[node] += degree

    # AVAREGE DEGREE
    average_degrees = {node: total / 50 for node, total in total_degrees.items()}
    print(average_degrees)

    # TOTAL AVAREGE DEGREE
    total_average_degree = sum(average_degrees.values()) / len(average_degrees)
    print(total_average_degree)

    # BASEMAP
    fig = plt.figure(figsize=(12, 9))
    m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines()
    m.fillcontinents(color='lightgray')
    m.drawmapboundary()
    m.drawcountries()
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True], linewidth=0.5, color='grey')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False], linewidth=0.5, color='grey')
    # HEATMAP
    colors = [(0, 0, degree/10) for node, degree in average_degrees.items()]
    max_degree = max(average_degrees.values())
    min_degree = min(average_degrees.values())
    cmap = plt.cm.Reds
    norm = plt.Normalize(min_degree, max_degree)

    for node, (lat, lon) in nodes_coordinates.items():
        x, y = m(lon, lat)
        degree = average_degrees[node]
        color = cmap(norm(degree))
        
        m.plot(x, y, 'o', color=color, markersize=10)           #size of the node on the map 

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])