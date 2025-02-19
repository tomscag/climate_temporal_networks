import numpy as np
import os 

### New Plot
from lib.plot.plot_heatmap import plot_heatmap
from lib.plot.draw_connectivity_earth_network import draw_connectivity_earth_network
from lib.plot.draw_tau_earth_network import draw_tau_earth_network   
from lib.plot.draw_variation_earth_network import draw_variation_earth_network
from lib.plot.plot_global_variations import plot_global_variations
from lib.plot.plot_tipping_elements import plot_tipping_elements
from lib.plot.plot_threshold_comparison import plot_threshold_comparison

import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
######################################
######################################
######################################

#%%

if __name__ == "__main__":

    #%% Set parameters
    savefig = True
    resfolder  = "./fig/"
    folderinput = "../Output/"
    filelist = ["tas_ssp1_2.6_model_awi_cm_1_1_mr_100_surr",
                "tas_ssp2_4.5_model_awi_cm_1_1_100_surr",
                "tas_ssp5_8.5_model_awi_cm_100_surr",
                "pr_ssp5_2.6_model_awi_cm_1_1_mr_100_surr",
                "pr_ssp5_8.5_model_awi_cm_100_surr",
                "era5_tas_100_surr",  # 5
                "tp_era5_1970_2021_100_surr",
                "tas_cmip6_historical_awi_cm_1_1_mr_100_surr"
                ]
    finput = folderinput + filelist[0]
    # baseline = np.arange(1970,1990)  # 2022,2042    1970,1990
    baseline = np.arange(2022,2042)
    
    variat_percnt = False
    lw_connectivity = True   # If True linewidth corresponds to connectivity (Lorenzo)

    # Analysis per decades
    # init_year = np.array([1970,1980,1990,2000,2010,2020])
    init_year = np.array([2040,2050,2060,2070,2080,2090,2100])
    # init_year = np.array([1970,1980,1990,2000,2010])
    #%% Create figure 10 years
    fig, axes = plt.subplots(2,3, figsize=(15,10), 
                              subplot_kw={'projection': ccrs.Robinson()})
    for idx, ax in enumerate(axes.flat[0:(len(init_year)-1)]): #(init_year[:-1]):
        print(f"Analyzing years: {init_year[idx]}-{init_year[idx+1]}")
        years = np.arange(init_year[idx]+1, init_year[idx+1]+1)
        plot = draw_variation_earth_network(ax, 
                                            finput, 
                                            years, 
                                            baseline, 
                                            variat_percnt, 
                                            lw_connectivity)
    # fig.delaxes(axes[1, 2])
    plt.tight_layout()
    
    # Save figure
    filename = "variat_percnt_" if variat_percnt else "variat_"
    filename += "width_connect_" if lw_connectivity else "width_variat_"
    filename += finput.split("Output/")[1]
    plt.savefig(f"{resfolder}{filename}.png",
                dpi=300,
                bbox_inches='tight')
    #%% Create figures 30 years
    
    init_year = np.array([2040,2070,2100])
    # init_year = np.array([1980,2000,2020])
    
    fig, axes = plt.subplots(2,1, figsize=(5,10), 
                              subplot_kw={'projection': ccrs.Robinson()})
    for idx, ax in enumerate(axes.flat[0:(len(init_year)-1)]): #(init_year[:-1]):
        print(f"Analyzing years: {init_year[idx]}-{init_year[idx+1]}")
        years = np.arange(init_year[idx]+1, init_year[idx+1]+1)
        plot = draw_variation_earth_network(ax, 
                                            finput, 
                                            years, 
                                            baseline, 
                                            variat_percnt, 
                                            lw_connectivity)
    plt.tight_layout()
    # Save figure
    filename = "variat_percnt_30yrs" if variat_percnt else "variat_"
    filename += "width_connect_" if lw_connectivity else "width_variat_"
    filename += finput.split("Output/")[1]
    plt.savefig(f"{os.path.join(resfolder,filename)}.png",
                dpi=300,
                bbox_inches='tight')
    
    
    #%% Plot tipping elements
    # fig, ax = plt.subplots(figsize=(25,10), subplot_kw={'projection':ccrs.Robinson()})
    
    # plot = plot_tipping_elements(ax)
    # plt.savefig(f"tipping_elements.png", dpi=400,
    #             bbox_inches='tight')



    #%% Plot threshold comparison
    # fig, axes = plt.subplots(3,1, figsize=(25,10), subplot_kw={'projection':ccrs.Robinson()})
    # years = np.arange(2040,2060)
    # # years = np.arange(1970,2021)
    # thresholds = [50, 95, 99]
    
    # for idx, ax in enumerate(axes.flat): #(init_year[:-1]):
    #     plot = plot_threshold_comparison(ax, finput, years, thresholds[idx])


    # # Save figure
    # filename = "threshold_comparison_"
    # filename += finput.split("Output/")[1]
    # plt.savefig(f"{os.path.join(resfolder,filename)}.png",
    #             dpi=300,
    #             bbox_inches='tight')
    
    
    #%%
    # from lib.plot.test import test
    # fig, ax = plt.subplots(figsize=(25,10), subplot_kw={'projection':ccrs.Robinson()})
    # plot = test(ax)
    
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


