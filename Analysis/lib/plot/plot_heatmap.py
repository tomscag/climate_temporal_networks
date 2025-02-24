import numpy as np
import matplotlib.pyplot as plt


from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature

from lib.misc import (
            total_degree_nodes,
            load_dataset_hdf5,
            sample_fuzzy_network,
            load_lon_lat_hdf5
            )

#############################
#############################
#############################



class plot_heatmap(PlotterEarth):

    def __init__(self,fnameinput, resfolder,years,nsamples,fname="heatmap_earth.png"):

        super().__init__()
        self.fname = fname
        self.fnameinput = fnameinput
        self.fnameoutput = "heatmap_" + fnameinput.split("Results_")[1]
        self.resfolder = resfolder
        self.years = years
        self.nsamples = nsamples
        self.lons, self.lats = load_lon_lat_hdf5(self.fnameinput)
        self.prb_mat = self.load_data()
        self.draw_heatmap()


    def load_data(self,index=2):
        # Index 0 is the zscore matrix, 1 for the tau, 2 for the probability

        # Average over the considered period
        for idx,year in enumerate(self.years):
            if idx==0:
                mat = load_dataset_hdf5(self.fnameinput,year,index)
            elif idx>0:
                mat += load_dataset_hdf5(self.fnameinput,year,index)
        mat /= len(self.years)
        return mat




    def draw_heatmap(self):

        nlevel = 10

        grid_lon, grid_lat = np.meshgrid(self.lons, self.lats)

        nlats, nlons = len(self.lats),len(self.lons)
        temp_weight = np.zeros(shape=(nlats,nlons)) # (37,72)

        for sample in range(self.nsamples):
            print(f"sample {sample}")
            graph = sample_fuzzy_network(self.prb_mat)
            temp_weight += total_degree_nodes(graph,self.lons,self.lats)

        self.weighted_node_degree = temp_weight/np.float(self.nsamples)

        # Define colormap and normalization
        cmap = plt.cm.rainbow
        norm = plt.Normalize(vmin=self.weighted_node_degree.min(), vmax=self.weighted_node_degree.max())  
        # norm = plt.Normalize(vmin=0.03, vmax=0.085)

        cs = self.ax.contourf(grid_lon, grid_lat, self.weighted_node_degree,nlevel,cmap=cmap,
                        transform=ccrs.PlateCarree(),norm=norm)
        # cs.set_clim(vmin=0.03, vmax=0.085)
        self.fig.colorbar(cs,location='right', label='Degree',aspect=10)

        # Show grid
        # self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
        #                 transform=ccrs.PlateCarree())

        plt.savefig(f"{self.resfolder}{self.fnameoutput}_{self.years[0]}_{self.years[-1]}.png",dpi=self.params['dpi'])