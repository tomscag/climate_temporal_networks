#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs

class plot_tipping_elements(PlotterEarth):

    def __init__(self, ax):
        super().__init__(ax)
        
        self.load_tipping_points()
                 
        self._draw_tipping_elements()
        
        
        
        
    def _draw_tipping_elements(self):
        
        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]       
            for lon, lat in coords:
                self.ax.plot(lat, lon, color=col, marker='o', markersize=8, alpha=0.85,
                             transform=ccrs.PlateCarree())