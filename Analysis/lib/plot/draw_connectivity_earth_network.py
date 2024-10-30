import numpy as np
import matplotlib.pyplot as plt
import ast

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from itertools import combinations
from lib.misc import (load_lon_lat_hdf5,
                      generate_coordinates,
                      compute_connectivity,
                      sample_fuzzy_network)

##################################
##################################
##################################


class draw_connectivity_earth_network(PlotterEarth):

    def __init__(self,
                 fnameinput: str, 
                 resfolder: str,
                 year: np.array):

        super().__init__()
        self.fnameinput = fnameinput
        
        self.resfolder = resfolder
        self.year = year
        self.set_title = True
        self.fnameoutput = self.set_fnameoutput()
        
        self.prb_mat = self.load_results(self.fnameinput,self.year,index=2)
        self.prb_mat = np.maximum(self.prb_mat,self.prb_mat.transpose())
        self.load_tipping_points()

        self.draw_connectivity_network()

    def set_fnameoutput(self):
        string = "connectivity_" 
        string += self.fnameinput.split("Results_")[1]
        return f"{self.resfolder}{string}_{self.year[0]}_{self.year[-1]}.png"


    def draw_connectivity_network(self):
        """
        # NOTE: we compute connectivity directly from the probability matrix
        #       since sampling is not needed in this case
        """
        lons, lats = load_lon_lat_hdf5(self.fnameinput)
        coords = generate_coordinates(5,lats,lons)
        coords = {tuple(val):key for key,val in coords.items()}

        ntip = len(self.tipping_points.keys())
        C = np.zeros(shape=(ntip,ntip))

        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]
                    c = compute_connectivity(self.prb_mat,coord1,coord2,coords)
                    C[id1,id2] += c
                    C[id2,id1] = C[id1,id2]


        # Draw connections between tipping elements
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:      
                    _,pos1 = self.tipping_centers[tip1]
                    _,pos2 = self.tipping_centers[tip2]    
                    self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=C[id1,id2]*20,
                                transform=ccrs.PlateCarree(),color="black",alpha=0.6)
                        
        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.scatter(coord[1], coord[0], color=col, 
                       s=100, label=name, transform=ccrs.PlateCarree())

        grid_lon, grid_lat = np.meshgrid(lons, lats)
        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
                        transform=ccrs.PlateCarree())
        
        if self.set_title:
            self.ax.set_title(f"Years: {self.year[0]} - {self.year[-1]}",
                              fontsize=30,weight='bold')

        plt.savefig(self.fnameoutput,
                    dpi=self.params['dpi'])