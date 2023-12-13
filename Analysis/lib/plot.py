# Plot functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from lib.misc import generate_coordinates


class PlotterEarth():

    def __init__(self,proj,year,resfolder,rows=1,cols=1) -> None:
        """
            Initialize oject attributes and create figure
        """
        self.proj = proj
        self.year = str(year)
        self.resfolder = resfolder
        self.map = Basemap(projection=proj, lat_0=0, lon_0=0)

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
                                 ncols=cols
                                 )
        self.ax.set_title(str(year))
        self.plot_earth_outline()

    def plot_earth_outline(self):
        '''
            Draw coastlines and meridians/parallel
        '''
        self.map.drawcoastlines()
        self.map.fillcontinents(color='gray',lake_color='gray',alpha=0.45)
        self.map.drawmapboundary()
        self.map.drawcountries()
        self.map.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True], linewidth=0.5, color='grey')
        self.map.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False], linewidth=0.5, color='grey')


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

        lats = np.arange(-90,90+5,5,dtype=float)  # 37 
        lons = np.arange(-180,180,5,dtype=float)         # 72

        lon2, lat2 = np.meshgrid(lons, lats)
        x, y = self.map(lon2,lat2) # Convert to meters

        cmap = plt.cm.viridis
        min_limit = 0.0 
        max_limit = 0.05
        norm = plt.Normalize(vmin=min_limit, vmax=max_limit)  # The limits of the colorbar

        cs = self.map.contourf(x,y,data,cmap=cmap,norm=norm)
        self.map.colorbar(location='right', label='Degree')
        plt.savefig(fname)

def plot_earth(data,sizegrid=5,proj="robin",savefig=False):
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
    
    # m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
    m = Basemap(projection=proj, lat_0=0, lon_0=0) # If you shift you have to shift lon lat too
    fig, ax = plt.subplots(figsize=(12, 9))
    lats = np.arange(-90,90+5,5,dtype=float)  # 37 
    lons = np.arange(-180,180,5,dtype=float)         # 72

    lon2, lat2 = np.meshgrid(lons, lats)
    # data = np.random.rand(len(lats),len(lons))
    x, y = m(lon2,lat2) # Convert to meters

    cmap = plt.cm.viridis
    min_limit = 0.0 
    max_limit = 0.05
    norm = plt.Normalize(vmin=min_limit, vmax=max_limit)        #choose the limit of the colorbar

    cs = m.contourf(x,y,data,cmap=cmap,norm=norm)

    
    m.drawcoastlines()
    m.fillcontinents(color='gray',lake_color='gray',alpha=0.45)
    m.drawmapboundary()
    m.drawcountries()
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[False, False, False, True], linewidth=0.5, color='grey')
    m.drawparallels(np.arange(-90., 91., 30.), labels=[True, False, False, False], linewidth=0.5, color='grey')
    # m.colorbar(location='right', label='Degree')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm,shrink=0.5, label="Area degree",aspect=10)
    # plt.savefig("test.png")



def plot_earth_lines(data):

    plot_earth(data,sizegrid=5,proj="robin")
