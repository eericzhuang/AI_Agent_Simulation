import torch
import torch.nn as nn 

class Model(nn.Module): 
    def __init__(self, n_input_features): 
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x): 
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
# construct model
model = Model(n_input_features=6)

# model name
FILE = "model.pth"

# lazy method
torch.save(model.state_dict(), FILE)
model = torch.load(FILE)
model.eval()

# recommended
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()