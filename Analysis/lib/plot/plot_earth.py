# Plot functions
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import glob
import ast

from lib.misc import (
            load_dataset_hdf5
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
    def __init__(self, ax) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.proj = ccrs.Robinson() # Earth projection "robin"
        # self.proj = ccrs.PlateCarree() # Earth projection "robin"
          
        # initialize figure as subplots
        # self.fig = plt.figure(figsize=(11, 8.5))
        # self.ax = plt.subplot(1, 1, 1, projection=self.proj)
        self.ax = ax

        # Set the axes using the specified map projection
        # self.ax=plt.axes(projection=self.proj)
        
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
        
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.OCEAN, facecolor=(0.8,0.8,0.8))
        self.ax.set_extent([-180, 180, -90, 90])
        
        # self.ax.set_facecolor(cfeature.COLORS['water'])
        # self.ax.add_feature(cfeature.LAND)
        


    def load_tipping_points(self):
        with open("../data/tipping_elements/tipping_points_positions_5deg.dat", 'r') as file:
            data = file.read()
        with open("../data/tipping_elements/tipping_points_centers.dat", 'r') as file:
            cent = file.read()
        self.tipping_points = ast.literal_eval(data) 
        self.tipping_centers = ast.literal_eval(cent)



    @staticmethod
    def load_results(folderinput,years,index):
        # Index 0 is the zscore matrix, 1 for the tau, 2 for the probability

        # Average over the considered period
        for idx, year in enumerate(years):
            fnameinput = glob.glob(folderinput + f"/*_year_{year}_maxlag_150.hdf5")[0]
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








        

