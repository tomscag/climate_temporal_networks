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
    ind_years = extract_year_limits(data)
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


def extract_year_limits(data):
    '''
        Return the indices of the first day of the years
    '''
    doy = np.array(data['dayofyear']) 
    doy = np.where( doy == 1)[0]
    doy = np.append(doy,len(data["time"])+1) # Add the index for the end of the last year
    return doy


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


    
def compute_connectivity(adj_mat,coord1,coord2,coords):
    # Compute connectivity considering 
    # Earth's spherical geometry
    fact = 2*np.pi/360 # For conversion to rad
    CC = 0
    D = 0   # Normalization factor
    for c1 in coord1:
        label1 = coords[c1]
        for c2 in coord2:
            label2 = coords[c2]
            CC += adj_mat[label1,label2]*np.cos(c1[0]*fact)*np.cos(c2[0]*fact)
            D += (np.cos(c1[0]*fact)*np.cos(c2[0]*fact))
    return CC/D



def load_lon_lat_hdf5(finput):
    dset = h5py.File(finput,"r")
    lons, lats = dset["lon"][:], dset["lat"][:]
    if max(lons)>=355:  # (0,355) -> (-180, 175)
        lons = lons - 180
    return lons, lats

def load_dataset_hdf5(finput,year,index):
    # Index 0 is the zscore matrix, 1 for the tau, 2 for the probability
    dset = h5py.File(finput,"r")
    return dset[str(year)][:,:,index] 


def sample_fuzzy_network(arr):
    # arr: matrix of probabilities (upper triangular)
    # Return the adjacency matrix
    N = arr.shape[0]
    arrc = arr.copy()
    arrc[arrc < np.random.random(size=(N,N)) ] = 0    # Sample fuzzy
    # return nx.from_numpy_array(arr,edge_attr="weight")
    graph = ig.Graph.Weighted_Adjacency(arrc,mode="upper")
    return graph.get_adjacency()



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
        Output:
            ACM:    Area connectivity matrix
                Each entry is a node position, organized by lat x lon,
                whose value is the total area to which a node
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

    c = 0    # Normalization factor
    for key,value in coords.items():
        c += np.abs(np.cos(value[0]*2*np.pi/360))

    W = {key:value/c  for key,value in W.items()}
    ACM = np.array(list(W.values())).reshape(len(lats),len(lons))
    return ACM