import numpy as np
import glob

from lib.misc import (
                load_edgelist,
                sample_fuzzy_network,
                load_dataset_hdf5
                )


class AnalyzeFuzzyNetwork():

    def __init__(self, fnameinput, years) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.fnameinput = fnameinput
        self.resfolder = "./fig/"
        self.years = years
        
        self.medium_fontsize = 20
        
        # self.prb_mat = self.load_results(self.fnameinput, self.years, index=2)
        # self.prb_mat = np.maximum(self.prb_mat, self.prb_mat.transpose())
        # self.numyears = self.endyear - self.startyear + 1

        # self.numfuzzy = 1  # Number of fuzzy networks to generate

        # self.fnameoutput = self._set_fnameoutput()

    def plot_clustering_coefficient(self, 
                                    ax, 
                                    mode: str = "gcc") -> None:
        '''
        Plot global clustering coefficient

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.
        mode: string
            gcc for global clustering 
            avgl for average local clustering

        Returns
        -------
        None.

        '''
        self.clust = np.empty(len(self.years))
        
        for idx, year in enumerate(self.years):
            print(f"Analyze clustering - year {year}")
            finput = glob.glob(self.fnameinput + 
                               f"/*_year_{year}_maxlag_150.hdf5")[0]
            
            prb_mat = load_dataset_hdf5(finput, year, index=2)
            prb_mat = np.maximum(prb_mat, prb_mat.transpose())
            graph = sample_fuzzy_network(prb_mat)
            
            if mode.lower() == "avgl":
                self.clust[idx] = graph.transitivity_avglocal_undirected(mode="NaN") # Average of the local clustering coefficient
            elif mode.lower() == "gcc":
                self.clust[idx] = graph.transitivity_undirected(mode="NaN") # Global clustering coefficient
    
        ax.plot(self.years, self.clust)
        ax.set_ylabel("Clustering")
        
    
    def plot_modularity(self, ax) -> None:
        '''
        Plot the modularity score

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.modularity = np.empty(len(self.years))
        
        for idx, year in enumerate(self.years):
            print(f"Analyze clustering - year {year}")
            finput = glob.glob(self.fnameinput + 
                               f"/*_year_{year}_maxlag_150.hdf5")[0]
            
            prb_mat = load_dataset_hdf5(finput, year, index=2)
            prb_mat = np.maximum(prb_mat, prb_mat.transpose())
            graph = sample_fuzzy_network(prb_mat)
            
            
            # Detect communities using the Louvain method
            communities = graph.community_multilevel()
            
            # Compute modularity
            self.modularity[idx] = graph.modularity(communities.membership)
            
            
        ax.plot(self.years, self.modularity)
        ax.set_ylabel("Modularity")
        
        
    def plot_edge_density(self, ax) -> None:
        '''
        Plot the edge density

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.density = np.empty(len(self.years))
        
        for idx, year in enumerate(self.years):
            print(f"Analyze clustering - year {year}")
            finput = glob.glob(self.fnameinput + 
                               f"/*_year_{year}_maxlag_150.hdf5")[0]
            
            prb_mat = load_dataset_hdf5(finput, year, index=2)
            prb_mat = np.maximum(prb_mat, prb_mat.transpose())
            graph = sample_fuzzy_network(prb_mat)
            
        
            # Compute 
            nn = graph.vcount()
            self.density[idx] = 2*graph.ecount()/(nn*(nn-1))
            
            
        ax.plot(self.years, self.density)
        ax.set_ylabel("Edge density", fontsize=self.medium_fontsize)
