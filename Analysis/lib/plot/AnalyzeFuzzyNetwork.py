import numpy as np
import glob
import matplotlib.pyplot as plt

from lib.misc import (
                load_edgelist,
                sample_fuzzy_network,
                load_dataset_hdf5
                )


class AnalyzeFuzzyNetwork():

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
    
            
