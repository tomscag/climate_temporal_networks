import numpy as np

### New Plot
from lib.plot.plot_heatmap import plot_heatmap
from lib.plot.draw_connectivity_earth_network import draw_connectivity_earth_network
from lib.plot.draw_tau_earth_network import draw_tau_earth_network   
from lib.plot.draw_variation_earth_network import draw_variation_earth_network
from lib.plot.plot_global_variations import plot_global_variations

import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
######################################
######################################
######################################



if __name__ == "__main__":

    resfolder  = "./fig/"
    folderinput = "../Output/"
    filelist = ["tas_ssp1_2.6_model_awi_cm_1_1_mr_100_surr",
                "tas_ssp2_4.5_model_awi_cm_1_1_100_surr",
                "tas_ssp5_8.5_model_awi_cm_100_surr",
                "pr_ssp5_2.6_model_awi_cm_1_1_mr_100_surr",
                "pr_ssp5_8.5_model_awi_cm_100_surr"
                ]
    finput = folderinput + filelist[3]
    baseline = np.arange(2022,2042)  # 2022,2042    1970,1990
    
    savefig = True
    variat_percnt = True

    # Analysis per decades
    # init_year = np.array([1970,1980,1990,2000,2010,2020])
    init_year = np.array([2040,2050,2060,2070,2080,2090,2100])
    # init_year = np.array([2041,2070])
    
    # Create figure
    fig, axes = plt.subplots(2,3, figsize=(25,10), 
                             subplot_kw={'projection': ccrs.Robinson()})
    for idx, ax in enumerate(axes.flat):     #(init_year[:-1]):
        print(f"Analyzing years: {init_year[idx]}-{init_year[idx+1]}")
        years = np.arange(init_year[idx]+1, init_year[idx+1]+1)
        plot = draw_variation_earth_network(ax, 
                                            finput, years, baseline, variat_percnt)
        # draw_connectivity_earth_network(finput,resfolder,years,nsamples)
        # draw_tau_earth_network(finput,resfolder,years)

    # plt.show()
    
    # Save figure
    string = "variat_percnt_" if variat_percnt else "variat_"
    string += finput.split("Output/")[1]
    plt.savefig(f"{resfolder}{string}.png",
                dpi=300,
                bbox_inches='tight')

    # plt.show()
    # 
    # plot_global_variations(finput,resfolder,limit_years=[2040,2100],nsamples,baseline)


    ## Plot Global Variations
    # plot_global_variations(finput)

    # Plot heatmap
    # years = np.arange(2080,2090)
    # plot_heatmap(finput,resfolder,years,nsamples)




    ## Plot teleconnections
    # plot_teleconnections(plote,fname,node=[34,45],K=5000)
    


    #######################
    #   Network analysis  #
    #######################

    ## Plot clustering coefficient
    # plote = PlotterLines(folderinput,resfolder)
    # plote.plot_clustering_coefficient()


