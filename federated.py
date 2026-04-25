import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def split_data(num_clients=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    idx = np.random.permutation(len(dataset))
    chunks = np.array_split(idx, num_clients)
    loaders = []
    for chunk in chunks:
        sub = Subset(dataset, chunk)
        loaders.append(DataLoader(sub, batch_size=64, shuffle=True))
    return loaders

def local_train(model, loader):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for imgs, labels in loader:
        opt.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return model.state_dict(), total_loss / len(loader)

# FedAvg - average weights from all clients
def fedavg(global_model, client_weights):
    avg = {}
    for key in global_model.state_dict():
        avg[key] = torch.stack([w[key].float() for w in client_weights]).mean(0)
    global_model.load_state_dict(avg)
    return global_model

def test_accuracy(model, loader, dataset_size):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            correct += (model(imgs).argmax(1) == labels).sum().item()
    return 100 * correct / dataset_size

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST('./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# experimented with 3 and 10 clients, 5 gave best balance
NUM_CLIENTS = 5
ROUNDS = 5

print(f"Federated Learning | {NUM_CLIENTS} clients | {ROUNDS} rounds")
print("-" * 45)

global_model = CNN()
client_loaders = split_data(NUM_CLIENTS)
start = time.time()

for r in range(1, ROUNDS + 1):
    weights = []
    losses = []
    for i, loader in enumerate(client_loaders):
        local = CNN()
        local.load_state_dict(global_model.state_dict())
        w, l = local_train(local, loader)
        weights.append(w)
        losses.append(l)
    global_model = fedavg(global_model, weights)
    acc = test_accuracy(global_model, test_loader, len(test_data))
    avg_loss = sum(losses) / len(losses)
    print(f"Round {r}/{ROUNDS} | Avg Client Loss: {avg_loss:.4f} | Global Accuracy: {acc:.2f}%")

elapsed = time.time() - start
final_acc = test_accuracy(global_model, test_loader, len(test_data))

print(f"\nFederated Learning Results:")
print(f"  Test Accuracy : {final_acc:.2f}%")
print(f"  Training Time : {elapsed:.2f}s")

torch.save(global_model.state_dict(), 'federated_model.pth')