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
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.hid = nn.Linear(hidden_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.hid(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out