import numpy as np
import matplotlib.pyplot as plt


from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature

from lib.misc import (
            create_fuzzy_network, 
            total_degree_nodes,
            load_dataset_hdf5,
            sample_fuzzy_network
            )

#############################
#############################
#############################



class plot_heatmap(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year,fname="heatmap_earth.png"):

        super().__init__()
        self.fname = fname
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.load_data()
        self.construct()


    def load_data(self,K=2000):

        elist = load_dataset_hdf5(self.fnameinput)

        # Create the full network "weighted" with the edge-probabilities
        graph = sample_fuzzy_network(elist)

        self.data = total_degree_nodes(graph)
        

    def construct(self):

        nlevel = 5
        lats = np.arange(-90,90+5,5,dtype=float)  # 37 
        lons = np.arange(-180,180,5,dtype=float)         # 72

        grid_lon, grid_lat = np.meshgrid(lons, lats)

        # Define colormap and normalization
        cmap = plt.cm.rainbow
        # norm = plt.Normalize(vmin=data.min(), vmax=data.max())  
        norm = plt.Normalize(vmin=0.03, vmax=0.085)

        cs = self.ax.contourf(grid_lon, grid_lat, self.data,nlevel,cmap=cmap,
                        transform=ccrs.PlateCarree(),norm=norm)
        cs.set_clim(vmin=0.03, vmax=0.085)
        self.fig.colorbar(cs,location='right', label='Degree',aspect=10)

        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
                        transform=ccrs.PlateCarree())
        plt.savefig(f"{self.resfolder}heatmap_{self.year}.png",dpi=self.params['dpi'])