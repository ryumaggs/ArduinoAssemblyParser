import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from functools import wraps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from plotly import tools
import plotly.offline as of
import plotly.graph_objs as go

class bcolors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

