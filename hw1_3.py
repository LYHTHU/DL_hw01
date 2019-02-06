import torch
import torch.nn as nn
from plot_lib import set_default, show_scatterplot, plot_bases
from matplotlib.pyplot import plot, title, axis
from time import perf_counter


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

seed = 1008
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

X = torch.randn(100, 2)