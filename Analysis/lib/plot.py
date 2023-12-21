# Plot functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from cartopy import crs as ccrs, feature as cfeature
from lib.misc import generate_coordinates


class PlotterEarth():
    ''' 
        Plotting grid data over earth surface

        Input:
        --------
            data : a 2D array of shape (nlats,nlons) containing values to plot
            proj : projection

        Returns
        -------
            None
    '''
    def __init__(self,proj,year,resfolder) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.proj = proj
        self.year = str(year)
        self.resfolder = resfolder

        # misc. figure parameters
        self.params = {'linewidth': 1,
                       'mrkrsize': 10,
                       'opacity': 0.8,
                       'width': 850,
                       'length': 700,
                       'dpi': 300
                       }        
        
        # colors
        self.colors = {'blue':'#377eb8',
                       'red' : '#e41a1c',

                       }

        # font for figure labels and legend
        self.lab_dict = dict(family='Arial',
                             size=26,
                             color='black'
                             )  
              
        # font for number labeling on axes
        self.tick_dict = dict(family='Arial',
                              size=24,
                              color='black'
                              )      
          
        # initialize figure as subplots
        self.fig = plt.figure(figsize=(11, 8.5))
        # self.ax = plt.subplot(1, 1, 1, projection=proj)

        # Set the axes using the specified map projection
        self.ax=plt.axes(projection=proj)
        
        self.ax.set_title(str(year))
        self.plot_earth_outline()

    def plot_earth_outline(self):
        '''
            Draw coastlines and meridians/parallel
        '''
        
        self.ax.coastlines()
        #self.ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
        # self.ax.set_facecolor(cfeature.COLORS['water'])
        # self.ax.add_feature(cfeature.LAND)
        # self.ax.add_feature(cfeature.COASTLINE)

        # self.ax.add_feature(cfeature.BORDERS, linestyle='--')
        # self.ax.add_feature(cfeature.LAKES, alpha=0.5)
        # self.ax.add_feature(cfeature.STATES)
        # self.ax.add_feature(cfeature.RIVERS)

    def plot_teleconnections(self,graph,initnodelist,fname="teleconnections.png"):
        '''
            Draw on a map all the teleconnections in the graph object
            starting from the nodes stored in initnodelist
        '''

        for initnode in initnodelist:
            print(f"Drawing teleconnections for node {initnode}")
            endnodes = list(graph[initnode])
            coords, lons, lats = generate_coordinates(sizegrid=5)
            latinit, loninit =  coords[initnode]
            lats = [coords[item][0] for item in endnodes ]
            lons = [coords[item][1] for item in endnodes ]
            
            for edges in range(len(endnodes)):
                alpha = graph[initnode][endnodes[edges]]['prob']
                self.map.drawgreatcircle(lon1=loninit,lat1=latinit,lon2=lons[edges],lat2=lats[edges],
                                color=self.colors['blue'],linewidth=self.params['linewidth'],
                                alpha=alpha       
                                        )
        plt.savefig(f"{self.resfolder}telecon_{self.year}.png",dpi=self.params['dpi'])
        plt.close()

    def plot_heatmap(self,data,fname="heatmap_earth.png"):

        nlevel = 5
        lats = np.arange(-90,90+5,5,dtype=float)  # 37 
        lons = np.arange(-180,180,5,dtype=float)         # 72

        grid_lon, grid_lat = np.meshgrid(lons, lats)

        # Define colormap and normalization
        cmap = plt.cm.rainbow
        # norm = plt.Normalize(vmin=data.min(), vmax=data.max())  
        norm = plt.Normalize(vmin=0.03, vmax=0.085)

        cs = self.ax.contourf(grid_lon, grid_lat, data,nlevel,cmap=cmap,
                     transform=ccrs.PlateCarree(),norm=norm)
        cs.set_clim(vmin=0.03, vmax=0.085)
        self.fig.colorbar(cs,location='right', label='Degree',aspect=10)

        # Show grid
        self.ax.plot(grid_lon,grid_lat,'k.',markersize=2, alpha=0.75,
                     transform=ccrs.PlateCarree())
        plt.savefig(f"{self.resfolder}heatmap_{self.year}.png",dpi=self.params['dpi'])




from lib.misc import (
                load_edgelist,
                create_fuzzy_network
                )
import networkx as nx
import igraph as ig
import os

class PlotterLines():

    def __init__(self,folderinput,resfolder,rows=1,cols=1) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.folderinput = folderinput
        self.resfolder = resfolder
        self.startyear = 1970
        self.endyear  =  2021
        self.numyears = self.endyear - self.startyear + 1

        self.numfuzzy = 1  # Number of fuzzy networks to generate

        # misc. figure parameters
        self.params = {'linewidth': 1,
                       'mrkrsize': 10,
                       'opacity': 0.8,
                       'width': 850,
                       'length': 700,
                       'dpi': 300
                       }        
        
        # colors
        self.colors = {'blue':'#377eb8',
                       'red' : '#e41a1c',
                       }

        # font for figure labels and legend
        self.lab_dict = dict(family='Arial',
                             size=26,
                             color='black'
                             )  
              
        # font for number labeling on axes
        self.tick_dict = dict(family='Arial',
                              size=24,
                              color='black'
                              )      
          
        # initialize figure as subplots
        self.fig, self.ax = plt.subplots(nrows=rows,
                                 ncols=cols, figsize=(20, 10),
                                 )
        # self.self.ax.set_title(str(year))
        # self.load_data()

    def load_data(self,fnameinput):

        df = load_edgelist(fnameinput)
        return df
        
    def get_filename(self, year,filelist):
        # Return filename of a particular year
        for file in filelist:
            if os.path.isfile(self.folderinput+os.sep+file ):
                if str(year) in file:
                    return file
        raise ValueError("FILE NOT FOUND")


    def plot_clustering_coefficient(self):
        filelist = os.listdir(self.folderinput)
        self.clust = np.empty(self.numyears)
        for idx, year in enumerate(range(self.startyear,self.endyear+1)):
            print(f"Load results for year {year}")
            inputfname = self.get_filename(year,filelist)
            edgelist = self.load_data(self.folderinput+inputfname)
            graph = create_fuzzy_network(edgelist)
            graph = ig.Graph.from_networkx(graph)
            self.clust[idx]  = graph.transitivity_avglocal_undirected(mode="NaN")
            # self.clust[idx] = nx.average_clustering(graph)
            # self.clust[idx] = nx.transitivity(graph)
    
        plt.plot(self.clust)
        plt.savefig("test_clust.png")

