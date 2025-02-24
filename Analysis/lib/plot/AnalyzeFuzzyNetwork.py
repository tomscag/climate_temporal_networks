import numpy as np
import glob

from lib.misc import (
                load_edgelist,
                sample_fuzzy_network,
                load_dataset_hdf5
                )


class AnalyzeFuzzyNetwork():

    def __init__(self, 
                 fnameinput: str, 
                 years: np.array,
                 modes: list
                 ) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.fnameinput = fnameinput
        self.resfolder = "./fig/"
        self.years = years
        
        
        self.medium_fontsize = 20
        
        self.modes = modes
        self.initialize_modes()

        
    def plot(self, 
             axes):
        
        for idx, year in enumerate(self.years):
            print(f"Analyze - year {year}")
            finput = glob.glob(self.fnameinput + 
                               f"/*_year_{year}_maxlag_150.hdf5")[0]
            
            prb_mat = load_dataset_hdf5(finput, year, index=2)
            prb_mat = np.maximum(prb_mat, prb_mat.transpose())
            self.graph = sample_fuzzy_network(prb_mat)
            
            for mode in self.modes:
                if mode == "clustering":
                    self.clust[idx] = self.compute_clustering_coefficient(mode="gcc")
                elif mode == "modularity":
                    self.modularity[idx] = self.compute_modularity()
                elif mode == "edge_density":
                    self.density[idx] = self.compute_edge_density()
                else:
                    print("Type not recognized")
        
        for idx, mode in enumerate(self.modes):
            
            if mode == "clustering":
                axes[idx].plot(self.years, self.clust)
                axes[idx].set_ylabel("Clustering", fontsize=self.medium_fontsize)
            elif mode == "modularity":
                axes[idx].plot(self.years, self.modularity)
                axes[idx].set_ylabel("Modularity", fontsize=self.medium_fontsize)
            elif mode == "edge_density":
                axes[idx].plot(self.years, self.density)
                axes[idx].set_ylabel("Edge density", fontsize=self.medium_fontsize)
            else:
                raise ValueError("Type not recognized")
                
        return axes
            

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
    
            
    def initialize_modes(self):
        for mode in self.modes:
            if mode == "clustering":
                self.clust = np.empty(len(self.years))
            elif mode == "modularity":
                self.modularity = np.empty(len(self.years))
            elif mode == "edge_density":
                self.density = np.empty(len(self.years))
            else:
                raise ValueError("Type not recognized")
