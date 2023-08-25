### 

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from math import e 
import pandas as pd
import numpy as np
import os
import networkx as nx



#####################################



def load_edgelist(fpath):
    with open(fpath,"r") as file:
        for line in file.readlines():
            print(line)

    elist = 0
    return elist


def read_edgelist(fpath):
    G = nx.read_edgelist(fpath,delimiter="\t", nodetype=int)















#####################################
#####################################
#####################################

## Parameters

finpath = "./Analysis/Output/"

finname = "network_period0.txt"


if __name__ == "__main__":

    # load_edgelist(finpath+finname)
    read_edgelist(finpath+finname)


    pass