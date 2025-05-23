import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs
from itertools import combinations
from lib.misc import (load_lon_lat_hdf5,
                      load_results,
                      load_tipping_points,
                      generate_coordinates,
                      compute_connectivity,
                      compute_total_area)

##################################
##################################
##################################


class draw_variation_earth_network(PlotterEarth):

    def __init__(self,
                 ax,
                 filenames: list,
                 years: np.array,
                 baseline: list,
                 variat_percnt: bool,
                 lw_connectivity: bool,
                 ):
        """
        Parameters
        ----------
        ax: axes
            Axes to plot
        filenames : str
            List of the results' folder paths (hdf files)
            Each element refers to a different model
        years : np.array
            Time window to average the results.
        baseline : list
            Time window to caclulate the baseline.
        variat_percnt : bool
            If true links represents percentage variations wrt the baseline.
        lw_connectivity: bool
            If true the linewidth is link connectivity, else the variation
        """

        super().__init__(ax)
        self.filenames = filenames
        self.resfolder = "./fig/"
        self.years = years
        self.baseline = baseline
        self.variat_percnt = variat_percnt
        self.lw_connectivity = lw_connectivity
        self.set_title = True
        self.set_colorbar = True
        self.show_grid = False
        self.set_labels = False
        self.save_fig = False
        self.linewidth = 150    # tas 100, pr 150 4e5
        self.tipping_points, self.tipping_centers = load_tipping_points()
        self.cmap = plt.get_cmap("RdBu_r")
        
        self.labels = {'el_nino_basin': 'EL', 'AMOC': 'AM', 
                       'tibetan_plateau_snow_cover': 'TB', 'coral_reef': 'CR', 
                       'west_antarctic_ice_sheet': 'WA', 'wilkes_basin': 'WI', 
                       'SMOC_south': 'SM', 'nodi_amazzonia': 'AZ', 
                       'boreal_forest': 'BF', 'artic_seaice': 'AS', 
                        'greenland_ice_sheet': 'GR', 'permafrost': 'PF',
                       'sahel': 'SH'}
        
        # Set colorbar limits
        if self.variat_percnt:
            self.vmin, self.vmax = -0.4, 0.4    # 0.2 for pr;  -0.4, 0.4 for tas
        else:
            self.vmin, self.vmax = -0.04, 0.04  # 0.2 for tas - 0.05 pr and era5
            
        variat = self._average_over_models()
        
        string = "variat_percnt_" if self.variat_percnt else "variat_"
        string += self.filenames[0].split("Output/")[1]
        self.fnameoutput = f"{self.resfolder}{string}_{self.years[0]}_{self.years[-1]}.png"
    
        self._draw_variation_network(variat)
        
        
    def _average_over_models(self) -> np.array:
        """Average over multiple models"""
        variations = [
            self._compute_variation_wrt_baseline(
                load_results(f, self.years, index=2),
                load_results(f, self.baseline, index=2)
            )
            for f in self.filenames
        ]
        
        return sum(variations) / len(self.filenames)


    def _get_color(self, value):
        norm_value = np.clip((value - self.vmin) /
                             (self.vmax - self.vmin), 0, 1)
        return self.cmap(norm_value)
    
    def _compute_variation_wrt_baseline(self, 
                                        prb_mat: np.array,
                                        prb_mat_base: np.array,
                                        ) -> np.array:
        """
        Parameters
        ----------
        prb_mat: np.array
            fuzzy matrix
        prb_mat_base: np.array
            fuzzy matrix baseline
        """
        
        self.lons, self.lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5, self.lats, self.lons)
        norm_fact = compute_total_area(coords)

        ntip = len(self.tipping_points.keys())
        C1 = np.zeros(shape=(ntip, ntip))*np.nan
        C2 = np.zeros(shape=(ntip, ntip))*np.nan

        # Compute variation for each tipping point
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]
                    C1[id1, id2] = compute_connectivity(
                        prb_mat_base, norm_fact, coord1, coord2, coords)
                    C1[id2, id1] = C1[id1, id2]
                    C2[id1, id2] = compute_connectivity(
                        prb_mat, norm_fact, coord1, coord2, coords)
                    C2[id2, id1] = C2[id1, id2]
        self.connectivity = C1
        
        if self.variat_percnt:
            self.thresh = 0.01
            variat = (C2 - C1)/C1
            variat[ np.abs(variat) < self.thresh] = 0
        else:
            self.thresh = 0.001
            variat = C2 - C1
            variat[ np.abs(variat) < self.thresh] = 0
        return variat
        

    def _draw_variation_network(self, variat: np.array):
        
        """
        NOTE: we compute connectivity directly from the probability matrix
              since sampling is not needed in this case
        
        Parameters
        ----------
        variat: np.array
            Tipping points connectivity variations wrt baseline
    
        """

        # Draw connections between tipping elements
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if (id1 < id2) & (np.abs(variat[id1,id2]) > self.thresh):
                    if self.lw_connectivity: 
                        c = self.connectivity[id1, id2]/7.5 
                    else: 
                        c = variat[id1, id2]
                    _, pos1 = self.tipping_centers[tip1]
                    _, pos2 = self.tipping_centers[tip2]

                    color = self._get_color(variat[id1, id2])
                    self.ax.plot(
                        [pos1[1], pos2[1]], [pos1[0], pos2[0]],
                        linewidth=np.abs(c)*self.linewidth,
                        color=color, transform=ccrs.PlateCarree())    
        
        # Show grid
        if self.show_grid:
            grid_lon, grid_lat = np.meshgrid(self.lons, self.lats)
            self.ax.plot(grid_lon, grid_lat, 'k.', markersize=1, alpha=0.35,
                         transform=ccrs.PlateCarree())

        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.plot(coord[1], coord[0], color=col, marker='o', markersize=10, alpha=0.85,
                         transform=ccrs.PlateCarree())
            
        # Write tipping elements labels
        if self.set_labels:
            for name, coords in self.labels.items():
                col, coord = self.tipping_centers[name]
                self.ax.text(coord[1], coord[0], self.labels[name], 
                             fontsize=13, color="black", weight='bold', 
                             transform=ccrs.PlateCarree())

        # Set colorbar
        if self.set_colorbar:
            norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            cb = plt.colorbar(sm, ax=self.ax, 
                              orientation='horizontal', shrink = 0.8,
                              pad = 0.025, aspect = 30)
            label = ("% " if self.variat_percnt else "") + \
                "variation wrt baseline"
            cb.set_label(label, fontsize=20)

        if self.set_title:
            self.ax.set_title(f"{self.years[0]} - {self.years[-1]}",
                              fontsize=30, weight='bold')
            
        if self.save_fig:
            plt.savefig(self.fnameoutput,
                        dpi=self.params['dpi'],
                        bbox_inches='tight')
