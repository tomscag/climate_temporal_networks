import numpy as np
import matplotlib.pyplot as plt
import ast

from lib.plot.plot_earth import PlotterEarth


##################################
##################################
##################################


class plot_tipping_elements(PlotterEarth):

    def __init__(self,fnameinput, resfolder,year,fname="heatmap_earth.png"):

        super().__init__()
        self.fname = fname
        self.fnameinput = fnameinput
        self.resfolder = resfolder
        self.year = year

        self.load_data()
        self.load_tipping_points()
        self.construct()


    
    def load_tipping_points(self):
        with open("../data/tipping_points_positions_5deg.dat", 'r') as file:
            data = file.read()
        self.tipping_points = ast.literal_eval(data) 