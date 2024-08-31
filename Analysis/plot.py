import numpy as np

### New Plot
from lib.plot.plot_heatmap import plot_heatmap
from lib.plot.draw_connectivity_earth_network import draw_connectivity_earth_network
from lib.plot.draw_tau_earth_network import draw_tau_earth_network   
from lib.plot.draw_variation_earth_network import draw_variation_earth_network
from lib.plot.plot_global_variations import plot_global_variations


                                


######################################
######################################
######################################



if __name__ == "__main__":

    years   = np.arange(2090,2100)
    nsamples = 10       # Number of samples from fuzzy network
    folderinput = "./Output/"
    # finput = folderinput + f"Results_2_6_precipitation_awi_cm_1_1_mr.hdf5"
    finput = folderinput + f"Results_tas_ssp5_8.5_model_awi_cm_1_1_mr_prior_0.5.hdf5"
    
    resfolder  = "./fig/"

    savefig = True

    # # Plot tipping elements
    # draw_connectivity_earth_network(finput,resfolder,years,nsamples)
    # draw_variation_earth_network(finput,resfolder,years,nsamples)
    # draw_tau_earth_network(finput,resfolder,years,nsamples)

    # Analysis per decades
    # init_year = np.array([2040,2050,2060,2070,2080,2090,2100])
    # # init_year = np.array([1970,1980,1990,2000,2010,2020])
    # for id,yrs in enumerate(init_year[:-1]):
    #     print(f"Analyzing years: {init_year[id:id+2]}")
    #     years = init_year[id:id+2]
    #     # draw_connectivity_earth_network(finput,resfolder,years,nsamples)
    #     # draw_variation_earth_network(finput,resfolder,years,nsamples)
    #     # draw_tau_earth_network(finput,resfolder,years,nsamples)


    ## Plot Global Variations
    # plot_global_variations(finput)

    # Plot heatmap
    years = np.arange(2080,2090)
    plot_heatmap(finput,resfolder,years,nsamples)




    ## Plot teleconnections
    # plot_teleconnections(plote,fname,node=[34,45],K=5000)
    


    #######################
    #   Network analysis  #
    #######################

    ## Plot clustering coefficient
    # plote = PlotterLines(folderinput,resfolder)
    # plote.plot_clustering_coefficient()


