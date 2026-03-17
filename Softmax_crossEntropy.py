import torch
import torch.nn as nn 
import numpy as np

# softmax
def softmax(x): 
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy: ', outputs)

# pytorch softmax
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print('softmax numpy: ', outputs)

# cross entropy
def cross_entropy(actual, predicted): 
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(actual.shape[0])

y = np.array([1, 0, 0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

loss1 = cross_entropy(y, y_pred_good)
loss2 = cross_entropy(y, y_pred_bad)

print(f'Loss1 numpy: {loss1:.4f}')
print(f'Loss2 numpy: {loss2:.4f}')

# pytorch cross entropy
# 3 samples
y = torch.tensor([2, 0, 1])

# n_samples, n_classes
y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

criterion = nn.CrossEntropyLoss()

loss1 = criterion(y_pred_good, y)
loss2 = criterion(y_pred_bad, y)

print(f'Loss1 numpy: {loss1.item():.4f}')
print(f'Loss2 numpy: {loss2.item():.4f}')

# predictions
_, prediction1 = torch.max(y_pred_good, dim=1)
_, prediction2 = torch.max(y_pred_bad, dim=1)

print(f'good prediction: {prediction1}')
print(f'bad prediction: {prediction2}')