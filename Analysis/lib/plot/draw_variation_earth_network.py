import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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


class draw_variation_earth_network(PlotterEarth):

    def __init__(self,
                 fnameinput: str,
                 resfolder: str,
                 year: np.array,
                 baseline: list):

        super().__init__()
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year
        self.baseline = baseline
        self.variat_percnt = True # If true links are percentage variations wrt the baseline
        self.set_title = True
        self.fnameoutput = self.set_fnameoutput()

        # Set colormap parameters
        self.cmap = plt.get_cmap("RdBu_r")
        self.vmin = -0.1
        self.vmax = 0.1

        self.prb_mat = self.load_results(self.fnameinput,self.year,index=2)
        self.prb_mat = np.maximum(self.prb_mat,self.prb_mat.transpose())
        self.load_tipping_points()

        self.draw_variation_network()

    def get_color(self,value): 
        norm_value = np.clip((value - self.vmin) / (self.vmax - self.vmin), 0, 1)
        return self.cmap(norm_value)

    def set_fnameoutput(self):
        string = "variations_percnt_" if self.variat_percnt else "variations_"
        string += self.fnameinput.split("Results_")[1]
        return f"{self.resfolder}{string}_{self.year[0]}_{self.year[-1]}.png"


    def draw_variation_network(self): 
        """
        # NOTE: we compute connectivity directly from the probability matrix
        #       since sampling is not needed in this case
        """

        lons, lats = load_lon_lat_hdf5(self.fnameinput)
        coords = generate_coordinates(5,lats,lons)
        coords = {tuple(val):key for key,val in coords.items()}

        self.prb_mat_base = self.load_results(self.fnameinput,self.baseline,index=2)
        self.prb_mat_base = np.maximum(self.prb_mat_base,self.prb_mat_base.transpose())

        ntip = len(self.tipping_points.keys())
        C1 = np.zeros(shape=(ntip,ntip))*np.nan
        C2 = np.zeros(shape=(ntip,ntip))*np.nan
        
        # Draw variation wrt baseline
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]
                    C1[id1,id2] = compute_connectivity(self.prb_mat_base,coord1,coord2,coords)
                    C1[id2,id1] = C1[id1,id2]
                    C2[id1,id2] = compute_connectivity(self.prb_mat,coord1,coord2,coords)
                    C2[id2,id1] = C2[id1,id2]
        
        if self.variat_percnt:
            variat = (C2 - C1)/C1
        else:
            variat = C2 - C1

        # Draw connections between tipping elements

        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:      
                    _,pos1 = self.tipping_centers[tip1]
                    _,pos2 = self.tipping_centers[tip2]   

                    color = self.get_color(variat[id1,id2])
                    self.ax.plot(
                        [pos1[1],pos2[1]],[pos1[0],pos2[0]], 
                        linewidth=np.abs(C1[id1,id2])*30,
                        color=color,transform=ccrs.PlateCarree())                     
  
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.50,
                        transform=ccrs.PlateCarree())
                    
        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.plot(coord[1], coord[0],color=col, marker='o',markersize=10, alpha=0.85,
                transform=ccrs.PlateCarree())

        # Set colorbar
        norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        sm = plt.cm.ScalarMappable(cmap=self.cmap,norm=norm)
        cb = plt.colorbar(sm,orientation='horizontal',ax=self.ax)
        label = ("Percentage " if self.variat_percnt else "")+ "variation respect to baseline"
        cb.set_label(label,fontsize=20)

        if self.set_title:
            self.ax.set_title(f"Years: {self.year[0]} - {self.year[-1]}",
                              fontsize=30,weight='bold')
        plt.savefig(self.fnameoutput,
                    dpi=self.params['dpi'],
                    bbox_inches='tight')

