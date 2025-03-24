import numpy as np
import pandas as pd
import glob
import ast
import matplotlib.pyplot as plt
import seaborn as sns

from lib.misc import (
                load_edgelist,
                load_tipping_points,
                sample_fuzzy_network,
                load_dataset_hdf5,
                compute_connectivity,
                generate_coordinates,
                compute_total_area,
                load_lon_lat_hdf5
                )


class RankTippingElements:
    def __init__(self,
                 fnameinputs: list[str],
                 years: np.array,
                 baseline: np.array) -> None:
        
        self.fnameinputs = fnameinputs
        self.baseline = baseline        
        self.years = years
        self.models = ["awi_cm_1_1_mr", "mri_esm_2.0", "CESM2"]
        self.resfolder = "./fig/"
        
        self.tipping_points, self.tipping_centers = load_tipping_points()
        self.figuresize = (2.33,2.33)
        self.variat_percnt = False
        
        self.labels = {'el_nino_basin': 'EL', 'AMOC': 'AM', 
                       'tibetan_plateau_snow_cover': 'TB', 'coral_reef': 'CR', 
                       'west_antarctic_ice_sheet': 'WA', 'wilkes_basin': 'WI', 
                       'SMOC_south': 'SM', 'nodi_amazzonia': 'AZ', 
                       'boreal_forest': 'BF', 'artic_seaice': 'AS', 
                        'greenland_ice_sheet': 'GR', 'permafrost': 'PF',
                       'sahel': 'SH'}
        
        self.arr = np.empty(shape=(len(self.tipping_points.keys()),len(fnameinputs)))
        
        
    def bar_plot(self, savefig:bool = True):
        fig, ax = plt.subplots(1, 1, figsize=self.figuresize, 
                               tight_layout = {'pad': .3})
    
    
        lons, lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5, lats, lons)
        norm_fact = compute_total_area(coords)
        
        
        for (m, fnameinput) in enumerate(self.fnameinputs):
            print(fnameinput)
            self.prb_mat = self._load_results(fnameinput, self.years, index=2)
            self.prb_mat_base = self._load_results(
                fnameinput, self.baseline, index=2)
            self.prb_mat_base = np.maximum(
                self.prb_mat_base, self.prb_mat_base.transpose())
    
            ntip = len(self.tipping_points.keys())
            C1 = np.zeros(shape=(ntip, ntip))*np.nan
            C2 = np.zeros(shape=(ntip, ntip))*np.nan
        
            # Draw variation wrt baseline
            for id1, tip1 in enumerate(self.tipping_points.keys()):
                for id2, tip2 in enumerate(self.tipping_points.keys()):
                    if id1 < id2:
                        coord1 = self.tipping_points[tip1]
                        coord2 = self.tipping_points[tip2]
                        C1[id1, id2] = compute_connectivity(
                            self.prb_mat_base, norm_fact, coord1, coord2, coords)
                        C1[id2, id1] = C1[id1, id2]
                        C2[id1, id2] = compute_connectivity(
                            self.prb_mat, norm_fact, coord1, coord2, coords)
                        C2[id2, id1] = C2[id1, id2]
            if self.variat_percnt:
                self.thresh = 0.01
                variat = (C2 - C1)/C1
                variat[ np.abs(variat) < self.thresh] = 0
            else:
                self.thresh = 0.00
                variat = C2 - C1
                variat[ np.abs(variat) < self.thresh] = 0
            self.arr[:,m] = np.nansum(variat,axis=1)
                    
        # Add bar plot
        tot = np.mean(self.arr,axis=1)
        idx = np.argsort(tot)
        data = pd.DataFrame(self.arr[idx,:], 
                            columns=self.models, 
                            index=np.array(list(self.labels.values()))[idx]
                            ).T
        data.index.name = "models"

        sns.stripplot(data=data, orient="h", ax=ax ,s=6, alpha=0.85)
        sns.boxplot(data=data, orient="h", saturation=0.4, ax=ax)
        ax.set_xlabel("total variation")
        ax.set_xlim([-1.0,0])
    
        if savefig:
            plt.savefig("bar_plot.pdf")
    
    def _load_results(self, folderinput, years, index):
        # Index 0 is the zscore matrix, 1 for the tau, 2 for the probability

        # Average over the considered period
        for idx, year in enumerate(years):
            fnameinput = glob.glob(
                folderinput + f"/*_year_{year}_maxlag_150.hdf5")[0]
            if idx == 0:
                mat = load_dataset_hdf5(fnameinput, year, index)
            elif idx > 0:
                mat += load_dataset_hdf5(fnameinput, year, index)
        mat /= len(years)
        return mat
       
    

class PlotterNetworkMetrics:

    def __init__(self, 
                 fnameinputs: list[str], 
                 years: np.array
                 ) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.fnameinputs = fnameinputs
        self.resfolder = "./fig/"
        self.years = years
        self.figuresize = (3.5,2.33/1.33)
        
        self.medium_fontsize = 18
        self.lw = 1.1
        
        self.legend = ["awi_cm_1_1_mr", "mri_esm_2.0", "CESM2"]
        self.arr = np.empty(len(self.years))

        
    def plot(self, 
                        mode: str
             ):
        
        #/ Define figure
        fig, ax = plt.subplots(1, 1, figsize=self.figuresize, tight_layout = {'pad': .3})
        ax.set_ylabel(f"{mode}", fontsize=self.medium_fontsize)
        
        #/ Specify colors and markers
        colors = ["#1f77b4", "#d62728", "#2ca02c"]
        
        window_size = 6
        weights = np.ones(window_size) / window_size
        
        for (m, fnameinput) in enumerate(self.fnameinputs):
            for idx, year in enumerate(self.years):
                print(f"Analyze - year {year}")
                finput = glob.glob(fnameinput + 
                                   f"/*_year_{year}_maxlag_150.hdf5")[0]
                
                prb_mat = load_dataset_hdf5(finput, year, index=2)
                prb_mat = np.maximum(prb_mat, prb_mat.transpose())
                self.graph = sample_fuzzy_network(prb_mat)
                
                if mode == "Clustering":
                    self.arr[idx] = self.compute_clustering_coefficient(mode="gcc")
                elif mode == "Modularity":
                    self.arr[idx] = self.compute_modularity()
                elif mode == "Edge density":
                    self.arr[idx] = self.compute_edge_density()
                else:
                    print("Type not recognized")
            
            # sma = np.convolve(self.arr, weights, mode='same') # valid
            ax.plot(self.years, self.arr, linewidth=self.lw, 
                    color=colors[m], label=self.legend[m])
        ax.legend(fontsize=9, loc="upper right", handlelength=1, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
                
        return ax
            

    

    def compute_clustering_coefficient(self, 
                                    mode: str = "gcc") -> None:
        '''
        Plot global clustering coefficient

        Parameters
        ----------
        mode: string
            gcc for global clustering 
            avgl for average local clustering

        '''
            
        if mode.lower() == "avgl":
            clust = self.graph.transitivity_avglocal_undirected(mode="NaN") # Average of the local clustering coefficient
        elif mode.lower() == "gcc":
            clust = self.graph.transitivity_undirected(mode="NaN") # Global clustering coefficient
    
        return clust

        
    
    def compute_modularity(self) -> None:
        '''
        Plot the modularity score
        '''
        
        
        # Detect communities using the Louvain method
        communities = self.graph.community_multilevel()
        
        # Compute modularity
        return self.graph.modularity(communities.membership)
            
    
    def compute_edge_density(self) -> None:
        '''
        Plot the edge density
        '''
            
        # Compute 
        nn = self.graph.vcount()
        return 2*self.graph.ecount()/(nn*(nn-1))
    
            
