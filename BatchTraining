import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np 
import math

class WineDataset(Dataset): 
    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        mean = self.x.mean(dim = 0)
        std = self.x.std(dim = 0)
        self.x = (self.x - mean) / std
        self.y = torch.from_numpy(xy[:, [0]]).long() - 1
        self.y = self.y.view(-1)
        self.n_samples = xy.shape[0]
        self.n_features = self.x.shape[1]

    def __getitem__(self, index):
        # dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
    def get_n_features(self): 
        return self.n_features

full_dataset = WineDataset()

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_subset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_subset, batch_size=4, shuffle=False, num_workers=2)

# construct model
class LogisticRegression(nn.Module): 
    def __init__(self, n_input_features, n_classes): 
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, n_classes)

    def forward(self, x): 
        return self.linear(x)
    
model = LogisticRegression(full_dataset.get_n_features(), 3)

# loss and optimizer
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 100
total_samples = len(full_dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(num_epochs): 
    for i, (inputs, labels) in enumerate(train_loader): 
        # forward pass and loss
        y_predicted = model(inputs)
        loss = criterion(y_predicted, labels)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        if (i + 1) % 10 == 0: 
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, loss {loss.item():.4f}')

model.eval()
with torch.no_grad(): 
    n_correct = 0
    n_samples = 0
    for inputs, labels in test_loader: 
        outputs = model(inputs)
        _, predicted_class = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted_class == labels).sum().item()
    
    acc = n_correct / n_samples
    print(f'accuracy = {acc:.4f}')

