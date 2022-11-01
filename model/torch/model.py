import torch.nn as nn
import torch as pt
from torch.autograd import Variable
import torch.nn.functional as F

_ = pt.manual_seed(142)
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


# # number of features (len of X cols)
# input_dim = 4
# # number of hidden layers
# hidden_layers = 25
# # number of classes (unique of y)
# output_dim = 3
class Network(nn.Module):
  def __init__(self, input_dim, hidden_layers, output_dim):
    super(Network, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layers)
    self.linear2 = nn.Linear(hidden_layers, output_dim)

  def forward(self, x):
    x = torch.sigmoid(self.linear1(x))
    x = self.linear2(x)
    return x