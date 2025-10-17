import numpy as np
import matplotlib.pyplot as plt

from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature
from lib.misc import (load_lon_lat_hdf5,
                      load_results,
                      load_tipping_points,
                      generate_coordinates,
                      compute_connectivity,
                      compute_total_area,                      
                      )

##################################
##################################
##################################


class plot_lag_analysis(PlotterEarth):

    def __init__(self, fnameinput, resfolder, years):
        
        self.fnameinput = fnameinput        
        self.resfolder = resfolder
        self.years = years
        self.set_title = True

        # Set colormap parameters
        self.cmap = plt.get_cmap("gist_rainbow") 
        self.vmin = 0 # Lag minimum and maximum
        self.vmax = 100

        self.prb_mat = load_results(self.fnameinput,self.years,index=2)
        self.tau_mat = load_results(self.fnameinput,self.years,index=1) # Tau mat
        self.tipping_points, self.tipping_centers = load_tipping_points()
        self.fnameoutput = f"lags_{self.resfolder}_year_{self.years}.png"


    def _get_color_tau(self,value):      
        norm_value = np.clip((value - self.vmin) / (self.vmax - self.vmin), 0, 1)
        return self.cmap(norm_value)


    def _compute_average_tau(self,coord1,coord2,coords):
        tau = 0
        count = 0
        for c1 in coord1:
            label1 = coords[c1]
            for c2 in coord2:
                label2 = coords[c2]
                if self.prb_mat[label1, label2] > 0:
                    tau += self.tau_mat[label1,label2]
                    count +=1

        if count > 0:
            return tau/count
        else:
            return np.nan



    def draw_tau_network(self, ax):
        super().__init__(ax)
        lons, lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5,lats,lons)
        norm_fact = compute_total_area(coords)

        ntip = len(self.tipping_points.keys())
        C = np.zeros(shape=(ntip,ntip))
        tau = np.zeros(shape=(ntip,ntip))

        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    _,pos1 = self.tipping_centers[tip1]
                    _,pos2 = self.tipping_centers[tip2]
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]

                    tau[id1,id2] += self._compute_average_tau(coord1,coord2,coords)
                    C[id1,id2] += compute_connectivity(self.prb_mat,
                                                       norm_fact,
                                                       coord1,
                                                       coord2,
                                                       coords)

        # Draw connections between tipping elements
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:      
                    _,pos1 = self.tipping_centers[tip1]
                    _,pos2 = self.tipping_centers[tip2]    

                    color = self._get_color_tau(tau[id1,id2])
                    self.ax.plot([pos1[1],pos2[1]],[pos1[0],pos2[0]], linewidth=C[id1,id2]*20,
                        color=color,transform=ccrs.PlateCarree()) 


        grid_lon, grid_lat = np.meshgrid(lons, lats)
        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.60,
                        transform=ccrs.PlateCarree(), rasterized=True)

        # Draw tipping elements positions
        for name, coords in self.tipping_points.items():
            col, coord = self.tipping_centers[name]

            self.ax.scatter(coord[1], coord[0], color=col, 
                       s=100, label=name, transform=ccrs.PlateCarree())
            # self.ax.plot(coord[1], coord[0], color=col, markersize=80, alpha=0.75,
            #             label=name, transform=ccrs.PlateCarree())

        # Set colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        cb = plt.colorbar(sm, ax=self.ax, orientation='horizontal', shrink=0.7, pad=0.05)
        cb.ax.tick_params(labelsize=18)
        cb.set_label("Lag (days)",fontsize=30)
        # ticks_loc = cb.get_ticks().tolist()
        cb.set_ticks(cb.get_ticks().tolist())
        cb.set_ticklabels([str(int(np.round(item))) for item in cb.get_ticks()*(self.vmax-self.vmin) + self.vmin])

        if self.set_title:
            self.ax.set_title(f"{self.years[0]} - {self.years[-1]}",
                              fontsize=30, weight='bold')



    def plot_heatmap(self, ax):
    
        import seaborn as sns
        
        self.labels = {'el_nino_basin': 'EL', 'AMOC': 'AM', 
                       'tibetan_plateau_snow_cover': 'TB', 'coral_reef': 'CR', 
                       'west_antarctic_ice_sheet': 'WA', 'wilkes_basin': 'WI', 
                       'SMOC_south': 'SM', 'nodi_amazzonia': 'AZ', 
                       'boreal_forest': 'BF', 'artic_seaice': 'AS', 
                        'greenland_ice_sheet': 'GR', 'permafrost': 'PF',
                       'sahel': 'SH'}
        
        lons, lats = load_lon_lat_hdf5()
        coords = generate_coordinates(5,lats,lons)
        ntip = len(self.tipping_points.keys())
        tau = np.zeros(shape=(ntip,ntip))    
    
        for id1, tip1 in enumerate(self.tipping_points.keys()):
            for id2, tip2 in enumerate(self.tipping_points.keys()):
                if id1 < id2:
                    _,pos1 = self.tipping_centers[tip1]
                    _,pos2 = self.tipping_centers[tip2]
                    coord1 = self.tipping_points[tip1]
                    coord2 = self.tipping_points[tip2]

                    tau[id1,id2] += self._compute_average_tau(coord1,coord2,coords)
                    
        sns.heatmap(tau, 
                    vmin=self.vmin, 
                    vmax=self.vmax, 
                    cmap="Reds", 
                    annot=True,
                    yticklabels=self.labels.values(),
                    xticklabels=self.labels.values(),
                    annot_kws={"fontsize":14})

        sns.set(font_scale=2)

