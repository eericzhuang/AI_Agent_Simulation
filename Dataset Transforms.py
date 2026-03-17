import torch
import torch.nn as nn 
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math

# data loading
class WineDataset(Dataset): 
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 2:]
        self.input_size = self.x.shape[1]
        self.y = xy[:, [1]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform: 
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return self.n_samples
    
    def get_input_size(self): 
        return self.input_size

# transforms   
class ToTensor(): 
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform(): 
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample): 
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

class Normalize(): 
    def __init__(self, x_mean, x_std, y_mean, y_std): 
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
        self.x_std = torch.tensor(x_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.tensor(y_std, dtype=torch.float32)
    
    def __call__(self, sample): 
        inputs, targets = sample
        inputs = (inputs - self.x_mean) / (self.x_std + 1e-7)
        targets = (targets - self.y_mean) / (self.y_std + 1e-7)
        return inputs, targets

# construct model  
class LinearRegression(nn.Module): 
    def __init__(self, input_size): 
        super(LinearRegression, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x): 
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.output(output)
        return output

# main function
if __name__ == "__main__": 
    # standardizing data
    raw_data = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)

    features = raw_data[:, 2:]
    x_mean = features.mean(axis=0)
    x_std = features.std(axis=0)

    raw_target = raw_data[:, [1]]
    y_mean = raw_target.mean(axis=0)
    y_std = raw_target.std(axis=0)

    # dataset transforms
    composed = torchvision.transforms.Compose([ToTensor(), Normalize(x_mean, x_std, y_mean, y_std)])
    dataset = WineDataset(transform=composed)

    # splitting dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_subset, test_subset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_subset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_subset, batch_size=4, shuffle=False, num_workers=2)

    # model
    model = LinearRegression(dataset.get_input_size())

    # loss and optimizer
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    num_epochs = 500
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4)

    # start training
    model.train()
    for epoch in range(num_epochs): 
        for i, (inputs, targets) in enumerate(train_loader): 
            # forward pass
            y_predicted = model(inputs)
            loss = criterion(y_predicted, targets)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # zero gradient
            optimizer.zero_grad()

            if (i + 1) % 15 == 0: 
                with torch.no_grad(): 
                    mae = torch.abs(y_predicted - targets).mean()
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i}/{n_iterations}, loss {loss.item():.4f}, mae {mae:.4f}')

    # start testing
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad(): 
        for inputs, targets in test_loader: 
            outputs = model(inputs)
            all_preds.append(outputs)
            all_targets.append(targets)

    # combine batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # inverse standardization
    all_preds = all_preds * y_std + y_mean
    all_targets = all_targets * y_std + y_mean

    # plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Regression Fitting')
    plt.grid(True)
    plt.show()