import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature

from lib.misc import (
            load_results,
            generate_coordinates,
            compute_total_area,
            compute_connectivity,
            total_degree_nodes,
            load_dataset_hdf5,
            load_lon_lat_hdf5,
            load_tipping_points
            )

#############################
#############################
#############################



class PlotterHeatmap:

    def __init__(self, 
                 ax,
                 fnameinput: str, 
                 years: np.array
                 ):

        super().__init__()
        self.fnameinput = fnameinput
        self.years = years
        self.cmap = "rocket"
        self.cmap_variat = "coolwarm"
        self.lons, self.lats = load_lon_lat_hdf5()
        
        
        self.prb_mat = load_results(fnameinput[0], self.years, index=2)

        self.tipping_points, self.tipping_centers = load_tipping_points()
        self.labels = {'EL': 'el_nino_basin', 'AM': 'AMOC', 
                       'TB': 'tibetan_plateau_snow_cover', 'CR': 'coral_reef', 
                       'WA': 'west_antarctic_ice_sheet', 'WI': 'wilkes_basin', 
                       'SM': 'SMOC_south', 'AZ': 'nodi_amazzonia', 
                       'BF': 'boreal_forest', 'AS': 'artic_seaice', 
                       'GR': 'greenland_ice_sheet', 'PF': 'permafrost', 
                       'SH': 'sahel'} 

        
        
    def draw_heatmap(self):


        self.lons, self.lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5, self.lats, self.lons)
        norm_fact = compute_total_area(coords)

        ntip = len(self.tipping_points.keys())
        C2 = np.zeros(shape=(ntip, ntip))*np.nan


        # Compute connectivity between each tipping elements
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]
                    # C1[id1, id2] = compute_connectivity(
                    #     prb_mat_base, norm_fact, coord1, coord2, coords)
                    # C1[id2, id1] = C1[id1, id2]
                    C2[id1, id2] = compute_connectivity(
                        self.prb_mat, norm_fact, coord1, coord2, coords)
                    C2[id2, id1] = C2[id1, id2]
        self.connectivity = C2

        sns.heatmap(self.connectivity, 
                    annot=True, fmt=".2f",
                    vmin=0,
                    vmax=0.3,
                    cmap=self.cmap, 
                    linewidths=0.5,
                    xticklabels=self.labels.keys(),
                    yticklabels=self.labels.keys(),
                    #cbar_kws={"orientation": "horizontal"}
                    )
        
        
    def draw_variation_heatmap(self, 
                               years_baseline: np.array, 
                               ):
        
        prb_mat_base = load_results(self.fnameinput[0], years_baseline, index=2)

        self.lons, self.lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5, self.lats, self.lons)
        norm_fact = compute_total_area(coords)

        ntip = len(self.tipping_points.keys())
        C1 = np.zeros(shape=(ntip, ntip))*np.nan
        C2 = np.zeros(shape=(ntip, ntip))*np.nan


        # Compute connectivity between each tipping elements
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]
                    C1[id1, id2] = compute_connectivity(
                        prb_mat_base, norm_fact, coord1, coord2, coords)
                    C1[id2, id1] = C1[id1, id2]
                    C2[id1, id2] = compute_connectivity(
                        self.prb_mat, norm_fact, coord1, coord2, coords)
                    C2[id2, id1] = C2[id1, id2]
        self.connectivity = C2
        
        self.thresh = 0.0
        variat = C2 - C1
        variat[ np.abs(variat) < self.thresh] = 0

        sns.heatmap(variat, 
                    annot=True, fmt=".2f",
                    vmin=-0.1,
                    vmax=0.1,
                    cmap=self.cmap_variat, 
                    linewidths=0.5,
                    xticklabels=self.labels.keys(),
                    yticklabels=self.labels.keys(),
                    #cbar_kws={"orientation": "horizontal"}
                    )


