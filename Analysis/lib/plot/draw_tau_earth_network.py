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


class draw_tau_earth_network(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year):

        super().__init__()
        self.fnameoutput = "lags"
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.adj_mat = self.load_results(self.fnameinput,self.year,index=2)
        self.load_tipping_points()

        self.draw_tau_network()


    def draw_tau_network(self):
        index = 1 # To load results
        lons, lats = load_lon_lat_hdf5(self.fnameinput)
        coords = generate_coordinates(5,lats,lons)
        coords = {tuple(val):key for key,val in coords.items()}
        self.tau_mat = self.load_results(self.fnameinput,self.year,index)

        cmap = plt.get_cmap("gist_rainbow")
        vmin, vmax = -90, 90 # Lag minimum and maximum

        def get_color_tau(value,cmap):      
            norm_value = np.clip((value - vmin) / (vmax - vmin), 0, 1)
            return cmap(norm_value)

        def compute_average_tau(adj_mat,coord1,coord2,coords):
            tau = 0
            for c1 in coord1:
                label1 = coords[c1]
                for c2 in coord2:
                    label2 = coords[c2]
                    tau += adj_mat[label1,label2]
            return tau/(len(coord1)*len(coord2))


        # Draw variation wrt baseline
        for tip1, tip2 in combinations(self.tipping_points.keys(),2):
            if tip1 != tip2:
                _,pos1 = self.tipping_centers[tip1]
                _,pos2 = self.tipping_centers[tip2]
                coord1 = self.tipping_points[tip1]
                coord2 = self.tipping_points[tip2]

                tau = compute_average_tau(self.tau_mat,coord1,coord2,coords)
                C = compute_connectivity(self.adj_mat,coord1,coord2,coords)
                # print(C)
                color = get_color_tau(tau,cmap)
                self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=C/200,
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
        sm = plt.cm.ScalarMappable(cmap=cmap)
        cb = plt.colorbar(sm,orientation='horizontal')
        cb.set_label("Average lag",fontsize=20)
        ticks_loc = cb.get_ticks().tolist()
        cb.set_ticks(cb.get_ticks().tolist())
        cb.set_ticklabels([str(np.round(item)) for item in cb.get_ticks()*(vmax-vmin) + vmin])


        plt.savefig(f"{self.resfolder}{self.fnameoutput}_{self.year}.png",dpi=self.params['dpi'])

