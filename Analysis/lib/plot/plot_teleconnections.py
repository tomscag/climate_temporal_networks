import numpy as np
import matplotlib.pyplot as plt


from lib.plot.plot_earth import PlotterEarth
from cartopy import crs as ccrs, feature as cfeature

from lib.misc import (
            create_fuzzy_network, 
            create_full_network,
            load_edgelist,
            filter_network_by_distance,
            total_degree_nodes
            )
from lib.misc import generate_coordinates

#############################
#############################
#############################

#TODO: fix, still not tested

class plot_teleconnections(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year,fname="teleconnections.png"):

        super().__init__()
        self.fname = fname
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.load_data()
        self.construct()




    def load_data(plote,fnameinput,node=[1050], K = 2000):

        elist = load_edgelist(fnameinput)

        # Filter out the small zscore
        elist = elist[ elist.zscore > 2.5]

        # Filter out short links
        elist = filter_network_by_distance(elist,K,filterpoles=True )

        # Create the full network "weighted" with the edge-probabilities
        graph = create_full_network(elist)

        ## Plot linemap
        plote.plot_teleconnections(graph,node)


    def construct(self,graph,initnodelist,fname="teleconnections.png"):
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