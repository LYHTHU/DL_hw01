import torch
import torch.nn as nn
# from plot_lib import set_default, show_scatterplot, plot_bases
from matplotlib.pyplot import plot, title, axis

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

seed = 1008
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)


def warm_up():
    raise NotImplementedError()

# Uncomment line below once you implement this function.
# print(warm_up())

def mul_row_loop(input_tensor):
    raise NotImplementedError()

from time import perf_counter
def times(input_tensor):
    raise NotImplementedError()

# Uncomment lines below once you implement this function.
# input_tensor = TODO
# time_1, time_2 = times(random_tensor)
# print('{}, {}'.format(time_1, time_2))