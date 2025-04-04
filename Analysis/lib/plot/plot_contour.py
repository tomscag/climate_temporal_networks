import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import xarray as xr

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from cartopy.util import add_cyclic_point

from lib.misc import (
            total_degree_nodes,
            load_dataset_hdf5,
            sample_fuzzy_network,
            load_lon_lat_hdf5,
            load_results
            )

#############################
#############################
#############################



class PlotterContour(PlotterEarth):

    def __init__(self, 
                 ax,
                 filenames: list, 
                 years: np.array,
                 baseline: np.array
                 ):

        super().__init__(ax)
        self.filenames = filenames
        self.years = years
        self.baseline = baseline
        self.cmap = plt.cm.rainbow
        self.lons, self.lats = load_lon_lat_hdf5()
        self._draw_contour()


    def _draw_contour(self):

        grid_lon, grid_lat = np.meshgrid(self.lons, self.lats)
        
        aws_test = []
        aws_base = []
        for f in self.filenames:
            print(f)
            prb_mat = load_results(f, self.years, index=2)
            prb_mat_base = load_results(f, self.baseline, index=2)
            graph = sample_fuzzy_network(prb_mat)
            aws_test.append(total_degree_nodes(graph, self.lons, self.lats))
            
            graph_base = sample_fuzzy_network(prb_mat_base)
            aws_base.append(total_degree_nodes(graph_base, self.lons, self.lats))
        aws_test = sum(aws_test)/len(aws_test)
        aws_base = sum(aws_base)/len(aws_base)
        
        aws = aws_test - aws_base
        
        aws = xr.DataArray(aws, 
                     dims=("latitude", "longitude"), 
                     coords={"longitude": self.lons,
                             "latitude": self.lats})
        
        aws = self._add_cyclic_point_to_dataset(aws)
        
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.8, "pad" :.05, 
                       'aspect':40, 'label':'Area-weighted connectivity',
                       }
        # Define colormap and normalization
        norm = plt.Normalize(vmin=aws.min(), 
                             vmax=aws.max())  
        # norm = plt.Normalize(vmin=0.03, vmax=0.085)
        norm = TwoSlopeNorm(vmin=-0.09, vcenter=0, vmax=0.03)
        aws.plot.contourf(ax=self.ax, transform=ccrs.PlateCarree(), 
                             cmap='Spectral_r', levels=21, 
                             norm=norm,
                             #vmin=self.vmin, vmax=self.vmax,
                             cbar_kwargs=cbar_kwargs)

        # Show grid
        # self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
        #                 transform=ccrs.PlateCarree())


        
        
    def _add_cyclic_point_to_dataset(self, data: xr.DataArray) -> xr.DataArray:
        ''' 
            Return a new DataArray where longitude have cyclic points
            in order to avoid bound effects at 360 deg
        '''
        cyclic_data, cyclic_longitude = add_cyclic_point(data.values, coord=data['longitude'])
    
        # Create new coords that will be used in creation of new dataset
        # Replicate coords of existing dataset and replace longitude with cyclic longitude
        coords = {dim: data.coords[dim] for dim in data.dims}
        coords["longitude"] = cyclic_longitude
    
        return xr.DataArray(cyclic_data, dims=data.dims, coords=coords)