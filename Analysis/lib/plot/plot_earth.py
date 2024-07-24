# Plot functions
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import ast

from lib.misc import (
            create_fuzzy_network, 
            total_degree_nodes,
            load_dataset_hdf5,
            load_lon_lat_hdf5,
            sample_fuzzy_network
            )


class PlotterEarth():
    ''' 
        Plotting grid data over earth surface

        Input:
        --------
            data : a 2D array of shape (nlats,nlons) containing values to plot
            proj : projection

        Returns
        -------
            None
    '''
    def __init__(self) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.proj = ccrs.Robinson() # Earth projection "robin"

          
        # initialize figure as subplots
        self.fig = plt.figure(figsize=(11, 8.5))
        # self.ax = plt.subplot(1, 1, 1, projection=proj)

        # Set the axes using the specified map projection
        self.ax=plt.axes(projection=self.proj)
        
        # self.ax.set_title(str(year))
        self.plot_earth_outline()

        # misc. figure parameters
        self.params = {'linewidth': 1,
                       'mrkrsize': 10,
                       'opacity': 0.8,
                       'width': 850,
                       'length': 700,
                       'dpi': 300
                       } 



    def plot_earth_outline(self):
        '''
            Draw coastlines and meridians/parallel
        '''
        
        self.ax.coastlines()
        #self.ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
        # self.ax.set_facecolor(cfeature.COLORS['water'])
        # self.ax.add_feature(cfeature.LAND)
        # self.ax.add_feature(cfeature.COASTLINE)

        # self.ax.add_feature(cfeature.BORDERS, linestyle='--')
        # self.ax.add_feature(cfeature.LAKES, alpha=0.5)
        # self.ax.add_feature(cfeature.STATES)
        # self.ax.add_feature(cfeature.RIVERS)

    def load_tipping_points(self):
        with open("../data/tipping_elements/tipping_points_positions_5deg.dat", 'r') as file:
            data = file.read()
        with open("../data/tipping_elements/tipping_points_centers.dat", 'r') as file:
            cent = file.read()
        self.tipping_points = ast.literal_eval(data) 
        self.tipping_centers = ast.literal_eval(cent)



    @staticmethod
    def load_results(fnameinput,years,index):
        # Index 0 is the zscore matrix, 1 for the tau, 2 for the probability

        # Average over the considered period
        for idx,year in enumerate(years):
            if idx==0:
                mat = load_dataset_hdf5(fnameinput,year,index)
            elif idx>0:
                mat += load_dataset_hdf5(fnameinput,year,index)
        mat /= len(years)
        return mat
        
        # if index == 2:
        #     # Create the full network "weighted" with the edge-probabilities
        #     graph = sample_fuzzy_network(mat)
        #     return graph.get_adjacency()
        # elif index == 1: # tau:
        #     return mat
        # elif index == 0: # zscore
        #     return mat
        # else:
        #     print("Load results: index not recognized!")





from lib.misc import (
                load_edgelist,
                create_fuzzy_network
                )
import networkx as nx
import igraph as ig
import os

class PlotterLines():

    def __init__(self,folderinput,resfolder,rows=1,cols=1) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.folderinput = folderinput
        self.resfolder = resfolder
        self.startyear = 1970
        self.endyear  =  2021
        self.numyears = self.endyear - self.startyear + 1

        self.numfuzzy = 1  # Number of fuzzy networks to generate

        # misc. figure parameters
        self.params = {'linewidth': 1,
                       'mrkrsize': 10,
                       'opacity': 0.8,
                       'width': 850,
                       'length': 700,
                       'dpi': 300
                       }        
        
        # colors
        self.colors = {'blue':'#377eb8',
                       'red' : '#e41a1c',
                       }

        # font for figure labels and legend
        self.lab_dict = dict(family='Arial',
                             size=26,
                             color='black'
                             )  
              
        # font for number labeling on axes
        self.tick_dict = dict(family='Arial',
                              size=24,
                              color='black'
                              )      
          
        # initialize figure as subplots
        self.fig, self.ax = plt.subplots(nrows=rows,
                                 ncols=cols, figsize=(20, 10),
                                 )
        # self.self.ax.set_title(str(year))
        # self.load_results()

    def load_results(self,fnameinput):

        df = load_edgelist(fnameinput)
        return df
        
    def get_filename(self, year,filelist):
        # Return filename of a particular year
        for file in filelist:
            if os.path.isfile(self.folderinput+os.sep+file ):
                if str(year) in file:
                    return file
        raise ValueError("FILE NOT FOUND")


    def plot_clustering_coefficient(self):
        filelist = os.listdir(self.folderinput)
        self.clust = np.empty(self.numyears)
        for idx, year in enumerate(range(self.startyear,self.endyear+1)):
            print(f"Load results for year {year}")
            inputfname = self.get_filename(year,filelist)
            edgelist = self.load_results(self.folderinput+inputfname)
            graph = create_fuzzy_network(edgelist)
            graph = ig.Graph.from_networkx(graph)
            self.clust[idx]  = graph.transitivity_avglocal_undirected(mode="NaN")
            # self.clust[idx] = nx.average_clustering(graph)
            # self.clust[idx] = nx.transitivity(graph)
    
        plt.plot(self.clust)
        plt.savefig("test_clust.png")

