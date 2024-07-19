### New Plot
from lib.plot.plot_heatmap import plot_heatmap
from lib.plot.plot_tipping_elements import plot_tipping_elements
                                   



######################################
######################################
######################################



if __name__ == "__main__":

    year       = 2043
    var        = 't2m'
    # folderinput = f"./Output_cluster/historical_{var}_1970_2022/"  # /t2m   /plev_{plev}
    folderinput = "./Output/correlations/"
    finput = folderinput + f"Results_2_6_precipitation_awi_cm_1_1_mr.hdf5"
    resfolder  = "./fig/"

    savefig = True

    # Plot tipping elements
    plot_tipping_elements(finput,resfolder,year)
    
    
    # Plot heatmap
    # plot_heatmap(finput,resfolder,year)




    ## Plot teleconnections
    # plot_teleconnections(plote,fname,node=[34,45],K=5000)
    


    


    #######################
    #   Network analysis  #
    #######################

    ## Plot clustering coefficient
    # plote = PlotterLines(folderinput,resfolder)
    # plote.plot_clustering_coefficient()


