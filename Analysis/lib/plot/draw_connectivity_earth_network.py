import numpy as np
import matplotlib.pyplot as plt
import ast

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from itertools import product,combinations
from lib.misc import (load_lon_lat_hdf5,
                      generate_coordinates,
                      compute_connectivity)

##################################
##################################
##################################


class draw_connectivity_earth_network(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year):

        super().__init__()
        self.fnameout = "connectivity"
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.adj_mat = self.load_results(self.fnameinput,self.year,index=2)
        self.load_tipping_points()

        self.draw_connectivity_network()

    


    def draw_connectivity_network(self):

        lons, lats = load_lon_lat_hdf5(self.fnameinput)
        coords = generate_coordinates(5,lats,lons)
        coords = {tuple(val):key for key,val in coords.items()}


        # Draw connections between tipping elements
        for tip1, tip2 in combinations(self.tipping_points.keys(),2):
            if tip1 != tip2:
                _,pos1 = self.tipping_centers[tip1]
                _,pos2 = self.tipping_centers[tip2]
                coord1 = self.tipping_points[tip1]
                coord2 = self.tipping_points[tip2]

                C = compute_connectivity(self.adj_mat,coord1,coord2,coords)
                self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=C/200,
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
        
        plt.savefig(f"{self.resfolder}{self.fnameout}_{self.year}.png",dpi=self.params['dpi'])