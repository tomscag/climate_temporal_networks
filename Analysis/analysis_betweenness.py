import random
import networkx as nx
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math


#PARAMETERS
n_periods = 10
n_graphs = 100

# COORDINATES
nodes_coordinates = {}
with open(f'Output/network_period0.txt', 'r') as f:
    for line in f.readlines():
        parts = line.split()      
        nodeA, nodeB = parts[0], parts[1]
        latA, lonA = eval(parts[3]) 
        latB, lonB = eval(parts[4]) 
        nodes_coordinates[nodeA] = (latA, lonA)
        nodes_coordinates[nodeB] = (latB, lonB)
nodes_coordinates = dict(sorted(nodes_coordinates.items()))

# ANALYSIS OVER PERIODS
tot_av_deg = []
tot_stdev_deg = []
periods = []
periodsplot =[]
all_average_betweenness = []  # list of lists for violin plot
medians = []  
lower_quartiles = []  
upper_quartiles = []  

for i in range(n_periods):
    year = 1970 + i*5
    periods.append(year)
    periodsplot.append(year+2.5)


# ANALYSIS
sum_cos_latitudes = sum(math.cos(math.radians(lat)) for lat, _ in nodes_coordinates.values())


for p in range(n_periods):
    print(f'Analysing period {p}')
    edges = []

    with open(f'Output/network_period{p}.txt', 'r') as f:
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
        graphs.append(edge_list)        #graphs contains (n_graphs) edge lists


    total_betweenness = {node: 0 for node in nodes_coordinates.keys()}  # dictionary for betweenness

    for edge_list in graphs:
        G = nx.Graph()  
        G.add_edges_from(edge_list)  
        betweenness = nx.betweenness_centrality(G)  # calcola la betweenness centrality per questo grafo
    
        # Aggiungi i valori di betweenness centrality al totale
        for node, value in betweenness.items():
            total_betweenness[node] += value


    # AVAREGE BETWEENNESS
    average_betweenness = {node: total / n_graphs for node, total in total_betweenness.items()}

    # ARRAYs FOR VIOLIN PLOT
    betweenness_list = list(average_betweenness.values())
    all_average_betweenness.append(betweenness_list)
    medians.append(np.median(betweenness_list))
    lower_quartiles.append(np.percentile(betweenness_list, 25))
    upper_quartiles.append(np.percentile(betweenness_list, 75))

    # BASEMAP
    fig, ax = plt.subplots(figsize=(12, 9))
    m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)

    m.drawcoastlines()
    m.fillcontinents(color='lightgray')
    m.drawmapboundary()
    m.drawcountries()
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True], linewidth=0.5, color='grey')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False], linewidth=0.5, color='grey')

    # HEATMAP
    colors = [(0, 0, degree/10) for node, degree in average_betweenness.items()]
    max_degree = max(average_betweenness.values())
    min_degree = min(average_betweenness.values())
    cmap = plt.cm.Reds
    min_limit = 0.0040 
    max_limit = 0.0090

    norm = plt.Normalize(vmin=min_limit, vmax=max_limit)        #choose the limit of the colorbar
 
    for node, (lat, lon) in nodes_coordinates.items():
        x, y = m(lon, lat)
        degree = average_betweenness[node]
        color = cmap(norm(degree))
        
        m.plot(x, y, 's', color=color, markersize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sm.set_clim(vmin=min_limit, vmax=max_limit)  

    # COLOR BAR
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, cax=cax, label="Average Betweenness Centrality")
    ax.set_title(f"Average Betweenness Centrality for Node {periods[p]} - {periods[p]+5}", loc='center', y=1.05)  # y=1.05 sposta il titolo un po' più in alto.
    #plt.show()
    plt.savefig(f"Plot/Betweenness/Average Betweenness Centrality for Node {periods[p]} - {periods[p]+5}.png")
    plt.close()









plt.figure(figsize=(10,6))
violin_parts = plt.violinplot(all_average_betweenness, positions=periodsplot, showmeans=False, widths=4)

for part in violin_parts['bodies']:
    part.set_facecolor('blue')
    part.set_edgecolor('black')

plt.plot(periodsplot, medians, label='Median', color='red')
plt.scatter(periodsplot, medians, color='red', s=50, zorder=5)

# Quartiles
plt.scatter(periodsplot, lower_quartiles, label='Lower Quartile', color='green', s=50, zorder=5)
plt.scatter(periodsplot, upper_quartiles, label='Upper Quartile', color='purple', s=50, zorder=5)
#plt.errorbar(periods, tot_av_deg, yerr=tot_stdev_deg, marker='o', capsize=5, ecolor='red', color='red', linestyle='-')

plt.xlabel('Five years period')
plt.ylabel('Average Betweenness Distribution')
plt.title('Average Betweenness Distribution of the Nodes over Time')
plt.xticks(np.arange(1970, 2020 + 1, 5))
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig(f"Plot/Betweenness/Violin plot Betweenness.png")
plt.close()
