import numpy as np
import matplotlib.pyplot as plt
import ast

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from itertools import product,combinations
from lib.misc import (load_lon_lat_hdf5,
                      generate_coordinates,
                      compute_connectivity,
                      sample_fuzzy_network)

##################################
##################################
##################################


class draw_tau_earth_network(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year,nsamples):

        super().__init__()
        self.fnameoutput = "lags"
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year
        self.nsamples = nsamples

        # Set colormap parameters
        self.cmap = plt.get_cmap("gist_rainbow") 
        self.vmin = -90 # Lag minimum and maximum
        self.vmax = 90


        self.prb_mat = self.load_results(self.fnameinput,self.year,index=2)
        self.tau_mat = self.load_results(self.fnameinput,self.year,index=1) # Tau mat
        self.load_tipping_points()

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
        lons, lats = load_lon_lat_hdf5(self.fnameinput)
        coords = generate_coordinates(5,lats,lons)
        coords = {tuple(val):key for key,val in coords.items()}
        

        ntip = len(self.tipping_points.keys())
        C = np.zeros(shape=(ntip,ntip))
        tau = np.zeros(shape=(ntip,ntip))

        for sample in range(self.nsamples):
            print(f"sample {sample}")
            self.adj_mat = sample_fuzzy_network(self.prb_mat)
            for id1, tip1 in enumerate(self.tipping_points.keys()):
                for id2, tip2 in enumerate(self.tipping_points.keys()):
                    if id1 < id2:
                        _,pos1 = self.tipping_centers[tip1]
                        _,pos2 = self.tipping_centers[tip2]
                        coord1 = self.tipping_points[tip1]
                        coord2 = self.tipping_points[tip2]

                        tau[id1,id2] += self.compute_average_tau(coord1,coord2,coords)
                        C[id1,id2] += compute_connectivity(self.prb_mat,coord1,coord2,coords)
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
        cb = plt.colorbar(sm,orientation='horizontal')
        cb.set_label("Average lag",fontsize=20)
        ticks_loc = cb.get_ticks().tolist()
        cb.set_ticks(cb.get_ticks().tolist())
        cb.set_ticklabels([str(np.round(item)) for item in cb.get_ticks()*(self.vmax-self.vmin) + self.vmin])


        plt.savefig(f"{self.resfolder}{self.fnameoutput}_{self.year[0]}_{self.year[-1]}.png",dpi=self.params['dpi'])

