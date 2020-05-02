import sys
import time
import random as rnd
import threading as th

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


EXT          = '.png'
IMG_DIR      = 'imgs/'
ATK_DIR      = IMG_DIR  + 'attack/'
ATK_VID_DIR  = ATK_DIR  + 'videos/'
CONT_DIR     = IMG_DIR  + 'contagion/'
CONT_VID_DIR = CONT_DIR + 'videos/'
DATASET_FILE = 'datasets/fb-pages-tvshow.edges'
MAPPING_FILE = 'datasets/fb-pages-tvshow.nodes'
GRAPH_NAME   = 'American TV Shows Facebook pages'


# to prettify the output
RED    = '\033[91m'
BLUE   = '\033[94m'
GREEN  = '\033[32m'
YELLOW = '\033[93m'
WHITE  = '\033[97m'
PURPLE = '\033[95m'
RESET  = '\033[0m'


# utility for graphs
METRIC_COLOR = {
    'betweenness': 'black',
    'closeness': 'red',
    'pagerank': 'green',
    'clustering': 'purple',
    'hits': 'blue orange',
    'random': 'orange'
}

METRIC_COMPLEMENTAR = {
    'betweenness': 'green',
    'closeness': 'blue',
    'pagerank': 'red',
    'clustering': 'orange',
    'hits': 'red black',
    'random': 'black'
}