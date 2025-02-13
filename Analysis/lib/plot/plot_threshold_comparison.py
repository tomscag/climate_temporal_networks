#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point

from lib.misc import (load_lon_lat_hdf5,
                      generate_coordinates,
                      compute_connectivity
                      )

##################################
##################################
##################################

class plot_threshold_comparison(PlotterEarth):
    
    def __init__(self,
                 ax,
                 fnameinput: str,
                 years: np.array,
                 threshold,
                 ):
        
        super().__init__(ax)
        self.ax = ax
        self.fnameinput = fnameinput
        self.years = years
        self.threshold = threshold
        
        
        self.load_tipping_points()
        self.lons, self.lats = load_lon_lat_hdf5()
        self.coords = generate_coordinates(5, self.lats, self.lons)
        self.nn = len(self.coords)        
         
        self.prb_mat = self.load_results(self.fnameinput, self.years, index=2)
        self.prb_mat = self.prb_mat + self.prb_mat.T  # Fill the lower matrix
        self.prb_mat = np.where(np.isnan(self.prb_mat),0, self.prb_mat)
        self.zscore = self.load_results(self.fnameinput, self.years, index=0)
        self.zscore = self.zscore + self.zscore.T  # Fill the lower matrix
        self._filter_zscores()
        self._plot_threshold_comparison()
    
    def _plot_threshold_comparison(self):
        
        
        
        fact = 2*np.pi/360
        cos = [np.cos(value[0]*fact) for key,value in self.coords.items()]
        A = sum(cos)
        # cos = np.tile(cos,(self.nn,1))
        # degree_zsc = np.sum(self.zscore*cos, axis=0)/fact
        
        degree_zsc = np.sum(self.zscore * cos, axis=1)/A
        degree_prb = np.sum(self.prb_mat * cos, axis=1)/A
        

        diff = degree_zsc - degree_prb
        # diff = degree_zsc
        diff = diff.reshape(37,72)
        diff = xr.DataArray(diff, 
                     dims=("latitude", "longitude"), 
                     coords={"longitude": self.lons,
                             "latitude": self.lats})
        
        diff = self._add_cyclic_point_to_dataset(diff)
        cbar_kwargs = {'orientation':'horizontal', 'shrink':0.175, "pad" : .05, 
                       'aspect':40, 'label':'difference',
                       }
        diff.plot.contourf(ax=self.ax, transform=ccrs.PlateCarree(), 
                             cmap='RdBu_r', levels=21, cbar_kwargs=cbar_kwargs)
    
    
    def _filter_zscores(self):
        percentile = np.percentile(self.zscore, self.threshold)
        self.zscore = np.where(self.zscore > percentile, 1, 0)



    def _add_cyclic_point_to_dataset(self, data: xr.DataArray) -> xr.DataArray:

        # Generate data with cyclic point and generate longitude with cyclic point
        cyclic_data, cyclic_longitude = add_cyclic_point(data.values, coord=data['longitude'])
    
        # Create new coords that will be used in creation of new dataset
        # Replicate coords of existing dataset and replace longitude with cyclic longitude
        coords = {dim: data.coords[dim] for dim in data.dims}
        coords["longitude"] = cyclic_longitude
    
        new_ds = xr.DataArray(cyclic_data, dims=data.dims, coords=coords)
        return new_ds 