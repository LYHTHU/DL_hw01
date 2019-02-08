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


set_default()
X = torch.randn(100, 2)

colors = X[:, 0]
show_scatterplot(X, colors, title='X')
OI = torch.cat((torch.zeros(2, 2), torch.eye(2)))
plot_bases(OI)


class Linear_fc_relu(torch.nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(Linear_fc_relu, self).__init__()
        self.m1 = nn.Linear(in_size, h_size)
        self.m2 = nn.Linear(h_size, out_size)

    def forward(self, x):
        y = x.view(-1, 2)
        y = torch.relu(self.m1(x))
        y = self.m2(y)
        return y


class Linear_fc_sigmoid(torch.nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(Linear_fc_sigmoid, self).__init__()
        self.m1 = nn.Linear(in_size, h_size)
        self.m2 = nn.Linear(h_size, out_size)

    def forward(self, x):
        y = x.view(-1, 2)
        y = torch.sigmoid(self.m1(x))
        y = self.m2(y)
        return y


linear_fc_relu = Linear_fc_relu(2, 5, 2)
y1 = linear_fc_relu(X).detach()

show_scatterplot(y1, colors, title='Relu')
OI = torch.cat((torch.zeros(2, 2), torch.eye(2)))
plot_bases(OI)

linear_fc_sigmoid = Linear_fc_sigmoid(2, 5, 2)
y2 = linear_fc_sigmoid(X).detach()

show_scatterplot(y1, colors, title='Sigmoid')
OI = torch.cat((torch.zeros(2, 2), torch.eye(2)))
plot_bases(OI)

