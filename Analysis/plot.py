### New Plot
from lib.plot.plot_heatmap import plot_heatmap



######################################
######################################
######################################



if __name__ == "__main__":

    year       = 1970
    var        = 't2m'
    folderinput = f"./Output_cluster/historical_{var}_1970_2022/"  # /t2m   /plev_{plev}
    fnameinput = f"t2m_year_{year}_maxlag_150.csv"
    lag_bounds = [-10,20]
    resfolder  = "./fig/"

    savefig = True


    # Plot heatmap
    plot_heatmap(folderinput+fnameinput,resfolder,year)


    ## Plot teleconnections
    # plot_teleconnections(plote,fname,node=[34,45],K=5000)
    


    


    #######################
    #   Network analysis  #
    #######################

    ## Plot clustering coefficient
    # plote = PlotterLines(folderinput,resfolder)
    # plote.plot_clustering_coefficient()


