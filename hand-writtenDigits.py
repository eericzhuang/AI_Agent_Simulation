# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training loop (batch training)
# Model evaluation
# GPU support
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28 x 28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 

# glimpse at the dataset
example = iter(train_loader)
samples, labels = example.next()
print(samples.shape, labels.shape)

for i in range(6): 
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.tight_layout()
# plt.show()

# construct model
class NeuralNet(nn.Module): 
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x): 
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
model.train()
n_total_steps = len(train_loader)
for epoch in range(num_epochs): 
    for i, (images, labels) in enumerate(train_loader): 
        # 100, 1, 28, 28
        # 100, 784
        # reshape images
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0: 
            print(f'epoch {epoch + 1}/{num_epochs}, step {i}/{n_total_steps}, loss {loss.item():.4f}')

# test
model.eval()
with torch.no_grad(): 
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader: 
        # reshape images
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    acc = n_correct / n_samples * 100.0
    print(f'accuracy = {acc}')