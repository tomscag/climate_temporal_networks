import numpy as np
import matplotlib.pyplot as plt
import ast

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from itertools import product,combinations
from lib.misc import (load_lon_lat_hdf5,
                      generate_coordinates)

##################################
##################################
##################################


class plot_tipping_elements(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year,fname="heatmap_earth.png"):

        super().__init__()
        self.fname = fname
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.load_data()
        self.load_tipping_points()
        self.construct()


    
    def load_tipping_points(self):
        with open("../data/tipping_points_positions_5deg.dat", 'r') as file:
            data = file.read()
        with open("../data/tipping_points_centers.dat", 'r') as file:
            cent = file.read()
        self.tipping_points = ast.literal_eval(data) 
        self.tipping_centers = ast.literal_eval(cent)

    def compute_connectivity(self,tip1,tip2,coords):
        coord1 = self.tipping_points[tip1]
        coord2 = self.tipping_points[tip2]
        C = 0
        for c1 in coord1:
            label1 = coords[c1]
            for c2 in coord2:
                label2 = coords[c2]
                C += self.adj_mat[label1,label2]*np.cos(c1[0])*np.cos(c2[0])/(np.cos(c1[0])*np.cos(c2[0]))
        return C


    def construct(self):

        lons, lats = load_lon_lat_hdf5(self.fnameinput)

        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.scatter(coord[1], coord[0], color=col, 
                       s=40, label=name, transform=ccrs.Geodetic())

        # Draw connections between tipping elements
        for tip1, tip2 in combinations(self.tipping_points.keys(),2):
            if tip1 != tip2:
                _,pos1 = self.tipping_centers[tip1]
                _,pos2 = self.tipping_centers[tip2]
                coords = generate_coordinates(5,lats,lons)
                coords = {tuple(val):key for key,val in coords.items()}
                C = self.compute_connectivity(tip1,tip2,coords)
                print(C)
                self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=C/100,
                             transform=ccrs.PlateCarree(),color="tab:blue")
                             
        grid_lon, grid_lat = np.meshgrid(lons, lats)

        # Define colormap and normalization
        cmap = plt.cm.rainbow
        # norm = plt.Normalize(vmin=self.data.min(), vmax=self.data.max())  

        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
                        transform=ccrs.PlateCarree())
        
        self.ax.plot([0,100],[100,210],'r.',linewidth=20.0,transform=ccrs.PlateCarree())

        plt.savefig(f"{self.resfolder}tipping_{self.year}.png",dpi=self.params['dpi'])