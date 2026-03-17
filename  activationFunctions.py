import torch
import torch.nn as nn 
import torch.nn.functional as F

# option 1: use nn.Modules
class NeuralNet(nn.Module): 
    def __init__(self, input_size, hidden_size): 
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        #nn.Softmax
        #nn.Tanh
        #nn.LeakyReLU

    def forward(self, x): 
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2: use activations functions directly in the forward pass
class NeuralNet(nn.Module): 
    def __init__(self, input_size, hidden_size): 
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x): 
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out

        #torch.softmax
        #torch.tanh
        #F.relu
        #F.leaky_relu