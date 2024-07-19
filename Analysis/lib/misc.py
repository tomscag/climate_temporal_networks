# Miscellaneous 
import networkx as nx
import numpy as np
import pandas as pd
from itertools import product
from numba import jit
import math
import h5py
import igraph as ig

#############################

def import_dataset(fileinput,variable='t2m', filterpoles=False):
    
    from netCDF4 import Dataset
    '''
    OUTPUT
        data: (2d-array)
            data reshaped

        ind_years: (1d-array)
            indices 1 Jan

        nodes: (dict)
            label: (lat,lon)

        filterpoles: (bool)
            if true remove degenerate nodes on the poles
    '''

    data = Dataset(fileinput, 'r')
    ind_years = first_day_of_year_index(data)
    lats  = [float(item.data) for item in data.variables['lat'] ]       
    lons  = [float(item.data) for item in data.variables['lon'] ]      
    ind_nodes = list(product(range(data.variables['lat'].size),range(data.variables['lon'].size)))  # [(0,0),(0,1),(0,2)...]

    temp = data.variables[variable]
    data = np.array(temp).reshape( temp.shape[0],temp.shape[1]*temp.shape[2]) # time, lat * lon

    nodes = list(product(lats,lons)) # [(-90,-180),(-90,-175)...]
    

    if filterpoles:
        nodes = {key:value   for key,value in nodes.items()   if np.abs(value[0]) < 90   }
        nodes[0] = (-90.0,-180.0) # I mantain the same labels in data
        nodes[2663] = (90.0,175.0) # 
        nodes = dict(sorted(nodes.items()))
    return data, ind_years, nodes, ind_nodes


def first_day_of_year_index(data):
    '''
        Return the indices of the first day of the years
    '''
    doy = np.array(data['dayofyear']) 
    return np.where( doy == 1)[0]


@jit(nopython=True)
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


def generate_coordinates(sizegrid,lats,lons):
    '''
    Output:
        coords (dict):
            key is the node id, value is a list 
            in the format [lat,lon]
    '''
    # lats = np.arange(-90,90+sizegrid,sizegrid,dtype=float)  # 37 
    # lons = np.arange(-180,180,sizegrid,dtype=float)         # 72
    

    N = len(lons)*len(lats)
    coords = {key:None for key in range(N)}
    node = 0
    for lat in lats:
        for lon in lons:
            coords[node] = [lat,lon]
            node += 1
    return coords


def create_full_network(edgelist):
    ''''
        Generate full network from edgelist
    '''

    G = nx.from_pandas_edgelist(edgelist,source="node1",target="node2",edge_attr=True)
    G.add_nodes_from(range(2664)) # In case some nodes are missing in the edgelist

    return G

def load_lon_lat_hdf5(finput):
    dset = h5py.File(finput,"r")
    lons, lats = dset["lon"][:], dset["lat"][:]
    if max(lons)>=355:  # (0,355) -> (-180, 175)
        lons = lons - 180
    return lons, lats

def load_dataset_hdf5(finput,year):
    dset = h5py.File(finput,"r")
    return dset[str(year)][:,:,2] # Index 0 is the zscore matrix, 2 for the probability


def sample_fuzzy_network(arr):
    # arr: matrix of probabilities (upper triangular)
    N = arr.shape[0]
    arr[arr < np.random.random(size=(N,N)) ] = 0    # Sample fuzzy
    # return nx.from_numpy_array(arr,edge_attr="weight")
    return ig.Graph.Weighted_Adjacency(arr,mode="upper")



def create_fuzzy_network(edgelist,mode="networkx"):
    ''''
        Generate fuzzy network from edgelist
    '''
    
    rnd = np.random.rand(*edgelist['prob'].shape)
    edgelist = edgelist.loc[ rnd < edgelist['prob'] ]
    
    match mode:
        case "networkx":
            G = nx.from_pandas_edgelist(edgelist,source="node1",target="node2",edge_attr=True)
            G.add_nodes_from(range(2664)) # In case some nodes are missing in the edgelist
        case "igraph":
            pass
        case _:
            print("Unknown mode")

    return G


def load_edgelist(name,lag_bounds = [0,10]):
    df = pd.read_csv(name,sep="\t",header=None,
                     names=["node1","node2","zscore","maxlag","prob"]
                     )
    df[['node1','node2']] = df[['node1','node2']].astype(int)
    return df



def load_edgelist_csv(name,lag_bounds = [0,10]):
    df = pd.read_csv(name,sep="\t",header=None,
                     names=["node1","node2","zscore","maxlag","prob"]
                     )
    # df = df.where( (df["maxlag"] >= lag_bounds[0]) & (df["maxlag"] <= lag_bounds[1]) ).dropna()
    # df['prob'].loc[ (df['maxlag'] < lag_bounds[0]) & df['maxlag'] > lag_bounds[1] ] = 0
    df[['node1','node2']] = df[['node1','node2']].astype(int)
    return df



def filter_network_by_distance(edgelist,K,filterpoles=False):
    '''
        Filter edgelist putting to zero links with longer than a threshold

        Input:
            K:   threshold in kilometers
            filterpoles: if true deletes nodes with abs(latitude) = 90 (north/sud poles) 
    '''
    
    coords, lons, lats = generate_coordinates(sizegrid=5)
    if filterpoles:
        edgelist['flag'] = edgelist.apply(lambda x:
                                            (coords[x.node1][0] > -90) and (coords[x.node1][0] < 90) and 
                                            (coords[x.node2][0] > -90) and (coords[x.node2][0] < 90), axis=1
                                        )  
        edgelist = edgelist.loc[ edgelist.flag == True ]

    edgelist['dist'] = edgelist.apply(lambda x: 
                                      haversine_distance( coords[x.node1][0],coords[x.node1][1],
                                                         coords[x.node2][0],coords[x.node2][1]  ), axis=1
                                      )
    return edgelist.loc[ edgelist.dist > K ]




def total_degree_nodes(G,lons,lats):
    """
        G:      igraph network
        Return:
            weights_matrix
                Each entry is a node, organized by lat x lon,
                and represents the total area to which a node
                is linked to
    """
    sizegrid = 5
    
    coords = generate_coordinates(sizegrid,lons,lats)
    # Only degree
    # data_matrix1 = np.array(list(dict(sorted(G.degree())).values())).reshape(len(lats),len(lons))

    W = {key:None for key in sorted(G.get_vertex_dataframe().index)} # Weighted connectivity
    for node in G.get_vertex_dataframe().index:

        w=0
        for item in G.neighbors(node):
            w += np.abs(np.cos(coords[item][0]*2*np.pi/360)) # cos lat
        W[node] = w

    c = 0
    for key,value in coords.items():
        c += np.abs(np.cos(value[0]*2*np.pi/360))

    W = {key:value/c  for key,value in W.items()}
    weights_matrix = np.array(list(W.values())).reshape(len(lats),len(lons))
    return weights_matrix