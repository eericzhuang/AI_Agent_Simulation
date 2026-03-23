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

# construct optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

# create checkpoint
checkpoint = {
    "epoch": 90, 
    "model_state": model.state_dict(), 
    "optim_state": optimizer.state_dict()
}

# save checkpoint
torch.save(checkpoint, "checkpoint.pth")

# load current checkpoint
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])

print(optimizer.state_dict())