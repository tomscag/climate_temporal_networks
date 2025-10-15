import numpy as np
import matplotlib.pyplot as plt
import ast

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from itertools import product,combinations
from lib.misc import (load_lon_lat_hdf5,
                      load_results,
                      load_tipping_points,
                      generate_coordinates,
                      compute_connectivity,
                      compute_total_area,
                      sample_fuzzy_network
                      )

##################################
##################################
##################################


class draw_tau_earth_network(PlotterEarth):

    def __init__(self, ax, fnameinput, resfolder, year, nsamples):

        super().__init__(ax)
        self.fnameinput = fnameinput
        
        self.resfolder = resfolder
        self.year = year
        self.nsamples = nsamples
        self.set_title = True

        # Set colormap parameters
        self.cmap = plt.get_cmap("gist_rainbow") 
        self.vmin = -20 # Lag minimum and maximum
        self.vmax = 20


        self.prb_mat = load_results(self.fnameinput,self.year,index=2)
        self.tau_mat = load_results(self.fnameinput,self.year,index=1) # Tau mat
        self.tipping_points, self.tipping_centers = load_tipping_points()
        self.fnameoutput = f"lags_{self.resfolder}_year_{self.year}.png"
        self.draw_tau_network()


    def get_color_tau(self,value):      
        norm_value = np.clip((value - self.vmin) / (self.vmax - self.vmin), 0, 1)
        return self.cmap(norm_value)


    def compute_average_tau(self,coord1,coord2,coords):
        tau = 0
        for c1 in coord1:
            label1 = coords[c1]
            for c2 in coord2:
                label2 = coords[c2]
                tau += self.adj_mat[label1,label2]
        return tau/(len(coord1)*len(coord2))



    def draw_tau_network(self):
        lons, lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5,lats,lons)
        norm_fact = compute_total_area(coords)

        ntip = len(self.tipping_points.keys())
        C = np.zeros(shape=(ntip,ntip))
        tau = np.zeros(shape=(ntip,ntip))

        for sample in range(self.nsamples):
            print(f"sample {sample}")
            self.adj_mat = sample_fuzzy_network(self.prb_mat).get_adjacency()
            for id1, tip1 in enumerate(self.tipping_points.keys()):
                for id2, tip2 in enumerate(self.tipping_points.keys()):
                    if id1 < id2:
                        _,pos1 = self.tipping_centers[tip1]
                        _,pos2 = self.tipping_centers[tip2]
                        coord1 = self.tipping_points[tip1]
                        coord2 = self.tipping_points[tip2]

                        tau[id1,id2] += self.compute_average_tau(coord1,coord2,coords)
                        C[id1,id2] += compute_connectivity(self.adj_mat,
                                                           norm_fact,
                                                           coord1,
                                                           coord2,
                                                           coords)
                        # print(C)

        C /= self.nsamples 
        tau /= self.nsamples 

        # Draw connections between tipping elements
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:      
                    _,pos1 = self.tipping_centers[tip1]
                    _,pos2 = self.tipping_centers[tip2]    

                    color = self.get_color_tau(tau[id1,id2])
                    self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=C[id1,id2]*20,
                        color=color,transform=ccrs.PlateCarree()) 


        grid_lon, grid_lat = np.meshgrid(lons, lats)
        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.60,
                        transform=ccrs.PlateCarree())

        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.scatter(coord[1], coord[0], color=col, 
                       s=100, label=name, transform=ccrs.PlateCarree())
            # self.ax.plot(coord[1], coord[0], color=col, markersize=80, alpha=0.75,
            #             label=name, transform=ccrs.PlateCarree())

        # Set colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        cb = plt.colorbar(sm, ax=self.ax, orientation='horizontal')
        cb.set_label("Average lag",fontsize=20)
        ticks_loc = cb.get_ticks().tolist()
        cb.set_ticks(cb.get_ticks().tolist())
        cb.set_ticklabels([str(np.round(item)) for item in cb.get_ticks()*(self.vmax-self.vmin) + self.vmin])

        if self.set_title:
            self.ax.set_title(f"Years {self.year[0]}s",fontsize=30,weight='bold')

        

