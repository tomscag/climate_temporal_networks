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


class plot_tipping_elements(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year,fname="heatmap_earth.png"):

        super().__init__()
        self.fname = fname
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.adj_mat = self.load_data(self.fnameinput,self.year)
        self.load_tipping_points()

        # self.draw_connectivity_network()
        self.draw_variation_network()

    
    def load_tipping_points(self):
        with open("../data/tipping_points_positions_5deg.dat", 'r') as file:
            data = file.read()
        with open("../data/tipping_points_centers.dat", 'r') as file:
            cent = file.read()
        self.tipping_points = ast.literal_eval(data) 
        self.tipping_centers = ast.literal_eval(cent)

    
    def get_color(self,value):
        if value > 0:
            return 'tab:red'
        elif value < 0:
            return 'tab:blue'
        else:
            return 'black' 


    def draw_variation_network(self,baseline=[2022]):

        lons, lats = load_lon_lat_hdf5(self.fnameinput)
        coords = generate_coordinates(5,lats,lons)
        coords = {tuple(val):key for key,val in coords.items()}

        year = baseline[0] # TODO 
        adj_mat_base = self.load_data(self.fnameinput,year)

        # Draw variation wrt baseline
        for tip1, tip2 in combinations(self.tipping_points.keys(),2):
            if tip1 != tip2:
                _,pos1 = self.tipping_centers[tip1]
                _,pos2 = self.tipping_centers[tip2]
                coord1 = self.tipping_points[tip1]
                coord2 = self.tipping_points[tip2]

                C1 = compute_connectivity(adj_mat_base,coord1,coord2,coords)
                C2 = compute_connectivity(self.adj_mat,coord1,coord2,coords)
                variat = C1-C2 # 
                color = self.get_color(variat)
                self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=variat/50,
                    color=color,transform=ccrs.PlateCarree()) 
        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.scatter(coord[1], coord[0], color=col, 
                       s=100, label=name, transform=ccrs.PlateCarree())

        grid_lon, grid_lat = np.meshgrid(lons, lats)
        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
                        transform=ccrs.PlateCarree())

        plt.savefig(f"{self.resfolder}tipping_{self.year}.png",dpi=self.params['dpi'])



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
                             transform=ccrs.PlateCarree(),color="tab:blue")

        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.scatter(coord[1], coord[0], color=col, 
                       s=100, label=name, transform=ccrs.PlateCarree())

        grid_lon, grid_lat = np.meshgrid(lons, lats)
        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
                        transform=ccrs.PlateCarree())
        
        plt.savefig(f"{self.resfolder}tipping_{self.year}.png",dpi=self.params['dpi'])