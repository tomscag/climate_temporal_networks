
import networkx as nx
import igraph as ig

import numpy as np
import matplotlib.pyplot as plt


from lib.misc import (
                generate_coordinates, 
                load_edgelist, 
                haversine_distance, 
                filter_network_by_distance,
                create_fuzzy_network
                    )

##############################




def percolation_analysis(G,p):
    '''
        Input
            G: networkx instance of a graph
            p: fraction of nodes to remove
        Output
            Return the largest connected component after 
            removing the p percentile of the nodes
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

def main():
    year  = 2000
    K =  3000   # Threshold to filter out short link

    # fpath = f"./Output/prob_numpyarray_year{year}.npy"
    fpath = f"./Output/correlations/plev_750/temperature_press_750_year_{year}_maxlag_150.csv"

    # Load edgelist
    edgelist = load_edgelist(fpath)

    # Filter edgelist
    edgelist = filter_network_by_distance(edgelist,K)

    # Create network
    G = create_fuzzy_network(edgelist)


    # gcc = percolation_analysis(G,0.1) 


    plot_gcc(G)


if __name__ == "__main__":
    main()
