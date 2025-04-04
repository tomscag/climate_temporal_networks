# Miscellaneous 
import networkx as nx
import numpy as np
import pandas as pd
from itertools import product
from numba import jit
import math
import h5py
import igraph as ig
import glob
import ast

#############################

def import_dataset(fileinput,variable='t2m'):
    
    from netCDF4 import Dataset
    from datetime import datetime
    '''
    OUTPUT
        data: (2d-array)
            data reshaped

        date_vec: (datetime.datetime)
            indices 1 Jan

        nodes: (dict)
            label: (lat,lon)
    '''

    with Dataset(fileinput, 'r') as data:
        lats  = [float(item.data) for item in data.variables['latitude'] ]       
        lons  = [float(item.data) for item in data.variables['longitude'] ]      
        ind_nodes = list(product(range(data.variables['latitude'].size),range(data.variables['longitude'].size)))  # [(0,0),(0,1),(0,2)...]
    
        date_vec = [datetime.strptime(item,'%Y-%m-%d' ) for item in data['date']]
    
        temp = data.variables[variable]
        data = np.array(temp).reshape( temp.shape[0],temp.shape[1]*temp.shape[2]) # time, lat * lon
    
        nodes = list(product(lats,lons)) # [(-90,-180),(-90,-175)...]
        
    return data, date_vec, nodes, ind_nodes



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


def generate_coordinates(sizegrid,lats,lons) -> dict:
    '''
    Output:
        coords (dict):
            key is the tuple (lat,lon), value is the node label 
    '''
    # lats = np.arange(-90,90+sizegrid,sizegrid,dtype=float)  # 37 
    # lons = np.arange(-180,180,sizegrid,dtype=float)         # 72
    

    N = len(lons)*len(lats)
    coords = {key:None for key in range(N)}
    node = 0
    for lat in lats:
        for lon in lons:
            coords[node] = (lat,lon)
            node += 1
    return {val: key for key, val in coords.items()}


def create_full_network(edgelist):
    ''''
        Generate full network from edgelist
    '''

    G = nx.from_pandas_edgelist(edgelist,source="node1",target="node2",edge_attr=True)
    G.add_nodes_from(range(2664)) # In case some nodes are missing in the edgelist

    return G


def compute_total_area(coords: dict) -> float:
    '''
    Compute the normalization factor for the connectivity 
    This is proportional to the total area of the Earth
    '''
    norm = 0
    for key, value in coords.items():
        norm += np.cos(np.deg2rad(key[0]))

    return norm
    # return norm**2 (this should be used in principle)

def compute_connectivity(adj_mat: np.array,
                         norm: float,
                         coord1: dict,
                         coord2: dict,
                         coords) -> float:
    '''
        Compute connectivity between two tipping points
        correcting for Earth's spherical geometry
    '''
    
    deg2rad = 2*np.pi/360 # For conversion to rad
    C = 0
    for c1 in coord1:
        label1 = coords[c1]
        for c2 in coord2:
            label2 = coords[c2]
            C += adj_mat[label1,label2]*np.cos(c1[0]*deg2rad)*np.cos(c2[0]*deg2rad)
    if np.isnan(C):
        pass
    return C/norm


def load_lon_lat_hdf5():
    
    lons = np.arange(-180,180,5, dtype=float)
    lats = np.arange(90, -95, -5, dtype=float)
    return lons, lats


def load_results(folderinput: str, years: np.array, index: int) -> np.array:
    """
        Index 0 is the zscore matrix, 1 for the tau, 2 for the probability
    """
    # Average over the considered period
    for idx, year in enumerate(years):
        fnameinput = glob.glob(folderinput + f"/*_year_{year}_maxlag_150.hdf5")[0]
        if idx==0:
            mat = load_dataset_hdf5(fnameinput,year,index)
        elif idx>0:
            mat += load_dataset_hdf5(fnameinput,year,index)
    mat /= len(years)
    mat[np.isnan(mat)] = 0
    return np.maximum(mat, mat.T)
    
    # if index == 2:
    #     # Create the full network "weighted" with the edge-probabilities
    #     graph = sample_fuzzy_network(mat)
    #     return graph.get_adjacency()
    # elif index == 1: # tau:
    #     return mat
    # elif index == 0: # zscore
    #     return mat
    # else:
    #     print("Load results: index not recognized!")


def load_dataset_hdf5(finput,year,index):
    # Index 0 is the zscore matrix, 1 for the tau, 2 for the probability
    dset = h5py.File(finput,"r")
    return dset["results"][:,:,index] 


def load_tipping_points():
    with open("../data/tipping_elements/tipping_points_positions_5deg.dat", 'r') as file:
        data = file.read()
    with open("../data/tipping_elements/tipping_points_centers.dat", 'r') as file:
        cent = file.read()
    tipping_points = ast.literal_eval(data) 
    tipping_centers = ast.literal_eval(cent)
    return tipping_points, tipping_centers


def sample_fuzzy_network(arr: np.array) -> ig.Graph:
    '''
    Parameters
    ----------
    arr : np.array
        2D-array of probabilities (upper triangular).

    Returns
    -------
    graph : ig.Graph
        Graph object

    '''
    N = arr.shape[0]
    arrc = arr.copy()
    arrc[arrc < np.random.random(size=(N, N))] = 0    # Sample fuzzy
    # return nx.from_numpy_array(arr,edge_attr="weight")
    graph = ig.Graph.Weighted_Adjacency(arrc, mode="upper")
    return graph



def create_fuzzy_network(edgelist:pd.DataFrame,
                         mode="networkx"):
    ''''
        Sample fuzzy network from edgelist (DEPRECATED)
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


def filter_network_by_distance(edgelist: pd.DataFrame,
                               K: float = 2000,
                               filterpoles: bool = False):
    '''
    Filter edgelist putting to zero links longer than a threshold

    Parameters
    ----------
        K:   threshold in kilometers
        filterpoles: if true deletes nodes with abs(latitude) = 90 (north/sud poles) 
    '''

    coords, lons, lats = generate_coordinates(sizegrid=5)
    if filterpoles:
        edgelist['flag'] = edgelist.apply(lambda x:
                                          (coords[x.node1][0] > -90) and (coords[x.node1][0] < 90) and
                                          (coords[x.node2][0] > -90) and (coords[x.node2][0] < 90), axis=1
                                          )
        edgelist = edgelist.loc[edgelist.flag == True]

    edgelist['dist'] = edgelist.apply(lambda x:
                                      haversine_distance(coords[x.node1][0], coords[x.node1][1],
                                                         coords[x.node2][0], coords[x.node2][1]), axis=1
                                      )
    return edgelist.loc[edgelist.dist > K]




def total_degree_nodes(G,lons,lats):
    """
    Parameters
    ----------
    G : ig.graph
        igraph object.
    lons : np.array
        Longitudes.
    lats : np.array
        Latitudes.

    Returns
    -------
    ACM : np.array
        Area Weighted Connectivity.
        The total area a node is connected to, weighted by the surface
    """     
    sizegrid = 5
    
    coords = generate_coordinates(sizegrid,lons,lats)
    coords = {value:key for key,value in coords.items()}

    W = {key:None for key in sorted(G.get_vertex_dataframe().index)} # Weighted connectivity
    fact = 2*np.pi/360
    for node in G.get_vertex_dataframe().index:
        w=0
        for item in G.neighbors(node):
            w += np.abs(np.cos(coords[item][0]*fact)) # cos lat
        W[node] = w

    # Normalization
    Z = sum([np.cos(value[0]*fact) for key,value in coords.items()])
    
    W = {key:value/Z  for key,value in W.items()}
    AWC = np.array(list(W.values())).reshape(len(lats),len(lons))
    return AWC
