### New Plot
from cartopy import crs as ccrs, feature as cfeature
from lib.plot import  PlotterEarth, PlotterLines
from lib.misc import (
            create_fuzzy_network, 
            create_full_network,
            load_edgelist,
            filter_network_by_distance,
            total_degree_nodes
            )


######################################
######################################
######################################


def plot_heatmap(plote,fnameinput,K=2000):

    elist = load_edgelist(fnameinput)

    # Filter out short links
    # elist = filter_network_by_distance(elist,K,filterpoles=True )

    # Create the full network "weighted" with the edge-probabilities
    graph = create_fuzzy_network(elist)

    weights_matrix = total_degree_nodes(graph)
    
    plote.plot_heatmap(weights_matrix)


def plot_teleconnections(plote,fnameinput,node=[1050], K = 2000):

    elist = load_edgelist(fnameinput)

    # Filter out the small zscore
    elist = elist[ elist.zscore > 2.5]

    # Filter out short links
    elist = filter_network_by_distance(elist,K,filterpoles=True )

    # Create the full network "weighted" with the edge-probabilities
    graph = create_full_network(elist)

    ## Plot linemap
    plote.plot_teleconnections(graph,node)



###################################
###################################


if __name__ == "__main__":

    year       = 1970
    plev       = 1000    # Pressure level
    folderinput = f"./Output/correlations/t2m/"  # /t2m   /plev_{plev}
    fnameinput = f"year_{year}_maxlag_150.csv"
    lag_bounds = [-10,20]
    fname = folderinput + fnameinput
    resfolder  = "./fig/"
    proj = ccrs.Robinson()     # Earth projection "robin"
    savefig = True

    plote = PlotterEarth(proj,year,resfolder)

    ## Plot teleconnections
    plot_teleconnections(plote,fname,node=[34,45],K=5000)
    

    ## Plot heatmap
    # plot_heatmap(plote,fname)
    


    #######################
    #   Network analysis  #
    #######################

    ## Plot clustering coefficient
    # plote = PlotterLines(folderinput,resfolder)
    # plote.plot_clustering_coefficient()


