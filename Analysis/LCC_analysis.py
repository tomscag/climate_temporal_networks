
import networkx as nx
import igraph as ig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from netCDF4 import Dataset
####################



def compute_distances_5deg():
    '''
    OUTPUT
        dist: (2d-array)
            distance matrix between points

        nodes: (dict)
            label: (lat,lon)
    '''

    data = Dataset(f'../data/t2m/anomalies_t2m_1970_2022_5grid.nc', 'r')
    lat  = data.variables['lat']        
    lon  = data.variables['lon']            
    # temp = data.variables['t2m']
    # data = np.array(temp).reshape( temp.shape[0],temp.shape[1]*temp.shape[2]) # time, lat * lon
    
    count = 0
    nodes = dict()
    for item_lat in enumerate(lat):
        for item_lon in enumerate(lon):
            nodes[count] = (float(item_lat[1].data),float(item_lon[1].data))
            count += 1
    
    N = 2664
    dist = np.zeros((N,N))
    for i in range(2664):
        for j in range(i+1,N):
            dist[i,j] = haversine_distance(nodes[i][0],nodes[i][1],nodes[j][0],nodes[j][1])
    dist = dist + dist.T    

    return  dist, nodes


def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371 #avarege radius

    # degree to radiant
    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180

    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    # haversine formula
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance


def create_fuzzy_network(edgelist):
    # random_matrix = np.random.rand(*adj_fuzzy.shape)
    rnd = np.random.rand(*edgelist['prob'].shape)
    edgelist = edgelist.loc[ rnd < edgelist['prob'] ]
    



    # Filter matrix (put to zero points that are close)
    dist, nodes = compute_distances_5deg()
    K = 8500
    # adj = filter_network(adj,dist,K)

    # Filter
    edgelist['dist'] = edgelist.apply(lambda x: haversine_distance( nodes[x.node1][0],nodes[x.node1][1],nodes[x.node2][0],nodes[x.node2][1]  ), axis=1)
    edgelist = edgelist.loc[ edgelist.dist > K ]

    # Igraph
    # G = ig.Graph.Adjacency(adj,mode="upper",diag=False)

    # G = nx.Graph(adj)
    G = nx.from_pandas_edgelist(edgelist,source="node1",target="node2")
    # plot_adj(adj,K)
    return G


def filter_network(adj,dist,K):
    '''
        Filter network by putting to zero those links with a distance grater than a threshold
    '''
    N = 2664
    for i in range(N):
        for j in range(i+1,N):
            if dist[i,j] < K:
                adj[i,j] = 0
                adj[j,i] = 0

    return adj


def percolation_analysis(G,p):
    '''
        Return the first and the second size of the largest connected component
    '''

    # rnd = np.random.rand(1,len(G.nodes))

    chosen_nodes = [x for x in G.nodes if  np.random.random() < p]
    G.remove_nodes_from(chosen_nodes)

    # gcc = nx.connected_components(G)
    components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    if len(components) > 0:
        return components[0]
    else:
        return 0

def plot_gcc(G):

    N = len(G.nodes)
    plist =  np.linspace(0,1,25)
    gcc = np.zeros(len(plist))
    count = 0
    for p in plist:
        
        
        components = percolation_analysis(G.copy(),p)
        gcc[count] = components/N
        print(f'nodes removal frac: {p:.3f}, size of gcc: {gcc[count]:.3f}')
        count += 1

    plt.plot(plist,gcc,'-*')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.ylabel("S")
    plt.xlabel("Fraction of nodes removed")
    plt.savefig("test_gcc.png")


def plot_adj(adj,K):
    plt.imshow(adj)
    plt.title(f"Filtered at {K} km")
    plt.savefig(f"adj_{K}.png")





####################


year  = 1973
# fpath = f"./Output/prob_numpyarray_year{year}.npy"
fpath = f"./Output/year_{year}_maxlag_150.csv"

# adj_fuzzy = np.load(fpath)
edgelist = pd.read_csv(fpath,delimiter="\t",names=["node1","node2","zscore","maxlag","prob"])


G = create_fuzzy_network(edgelist)


# gcc = percolation_analysis(G,0.1) 


plot_gcc(G)



