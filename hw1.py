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
    M = torch.full((13, 13), 1)
    M[1] = 2
    M[6] = 2
    M[11] = 2
    M[:, 1] = 2
    M[:, 6] = 2
    M[:, 11] = 2
    M[3:5,3:5] = 3
    M[3:5, 8:10] = 3
    M[8:10, 3:5] = 3
    M[8:10, 8:10] = 3
    return M
    # raise NotImplementedError()
print(warm_up())

# Write a function mul_row_loop, using python loops (and not even slicing operators),
# that gets a 2D tensor as input, and returns a tensor of same size, equal to the one given as argument,
# with the first row kept unchanged, the second multiplied by two, the third by three, etc. For instance:
def mul_row_loop(input_tensor):
    ret = input_tensor
    for i in range(ret.size(0)):
        for j in range(ret.size(1)):
            ret[i][j] = (i+1)*ret[i][j]
    return ret
    # raise NotImplementedError()

t = torch.full((4, 8), 2.0)
print(mul_row_loop(t))

# Write a second version of the same function named mul_row_fast which uses tensor operations and no looping.
# Hint: Use broadcasting and torch.arange, torch.view, and torch.mul.
def mul_row_fast(input_tensor):
    raise NotImplementedError()

from time import perf_counter
def times(input_tensor):
    raise NotImplementedError()

# Uncomment lines below once you implement this function.
# input_tensor = TODO
# time_1, time_2 = times(random_tensor)
# print('{}, {}'.format(time_1, time_2))