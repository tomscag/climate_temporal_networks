import random
from netCDF4 import Dataset
import networkx as nx
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math


#PARAMETERS
n_periods = 10
n_graphs = 50





# DATA INPUT
data = Dataset('./data/t2m/filtered_t2m_1970_2022_4grid.nc', 'r')
lat = data.variables['lat'][:]        
lon = data.variables['lon'][:]            
#temp = data.variables['t2m']

# COORDINATES
nodes_coordinates = {}
with open(f'./Analysis/Output/network_period0.txt', 'r') as f:
    for line in f.readlines():
        parts = line.split()      
        nodeA, nodeB = parts[0], parts[1]
        Ai = int(parts[3][1:])
        Aj = int(parts[4][:-1])
        Bi = int(parts[5][1:])
        Bj = int(parts[6][:-1])  
        print(Ai)    
        print(lat[Ai])  
        nodes_coordinates[nodeA] = (lat[Ai], lon[Aj])
        nodes_coordinates[nodeB] = (lat[Bi], lon[Bj])
nodes_coordinates = dict(sorted(nodes_coordinates.items()))

print(nodes_coordinates)

# ANALYSIS OVER PERIODS
tot_av_deg = []
tot_stdev_deg = []
periods = []
for i in range(n_periods):
    year = 1970 + i*5
    periods.append(year)


# ANALYSIS
sum_cos_latitudes = sum(math.cos(math.degrees(lat)) for lat, _ in nodes_coordinates.values())


for p in range(n_periods):
    print(f'Analysing period {p}')
    edges = []

    with open(f'./Analysis/Output/network_period{p}.txt', 'r') as f:
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
        graphs.append(edge_list)        #graphs contains (n_graphs=50) edge lists


    total_degrees = {node: 0 for node in nodes_coordinates.keys()}      #dictionary for degrees (it takes the same keys from nodes_coordinates)
    #print(total_degrees)

    for edge_list in graphs:
        G = nx.Graph()  
        G.add_edges_from(edge_list)  

        for edge in edge_list:
            nodoA, nodoB = edge
            total_degrees[nodoA] += math.cos(math.degrees(nodes_coordinates[nodoA][0]))/sum_cos_latitudes
            total_degrees[nodoB] += math.cos(math.degrees(nodes_coordinates[nodoB][0]))/sum_cos_latitudes

    #print(total_degrees)




    # AVAREGE DEGREE
    average_degrees = {node: total / 50 for node, total in total_degrees.items()}
    #print(average_degrees)

    # TOTAL AVAREGE DEGREE and DEVIATION STANDARD
    total_average_degree = sum(average_degrees.values()) / len(average_degrees)
    tot_av_deg.append(total_average_degree)
    std_dev = np.std(list(average_degrees.values()))
    tot_stdev_deg.append(std_dev)

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
    colors = [(0, 0, degree/10) for node, degree in average_degrees.items()]
    max_degree = max(average_degrees.values())
    min_degree = min(average_degrees.values())
    cmap = plt.cm.Reds
    norm = plt.Normalize(min_degree, max_degree)

    for node, (lat, lon) in nodes_coordinates.items():
        x, y = m(lon, lat)
        degree = average_degrees[node]
        color = cmap(norm(degree))
        
        m.plot(x, y, 'o', color=color, markersize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # COLOR BAR
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, cax=cax, label="Avarege Weigthed Degree")

    ax.set_title("Avarege Weigthed Degree for Node", loc='center', y=1.05)  # y=1.05 sposta il titolo un po' più in alto.
    # plt.show()
    plt.savefig(f"./period_{p}.png")

    plt.close()

# PLOT TOT AV DEGREE OVER TIME
plt.errorbar(periods, tot_av_deg, yerr=tot_stdev_deg, marker='o', capsize=5, label='Grado medio', color='red', linestyle='-')
plt.xlabel('Five years period')
plt.ylabel('Avarege Tot Degree')
plt.title('Avarege Total Degrees over Time')
start, end = plt.xlim()  
plt.xticks(np.arange(1970, 2020 + 1, 5))
plt.ylim(0, 2)
plt.legend()  
plt.grid(True) 

# plt.show()
plt.savefig(f"./Avg_degree.png")

