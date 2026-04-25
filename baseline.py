import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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

# load dataset - using standard MNIST normalization values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()

# tried SGD first but Adam converged faster
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training baseline (centralized, no privacy)...")
start = time.time()

for epoch in range(5):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/5 | Loss: {avg_loss:.4f}")

elapsed = time.time() - start

# evaluate on test set
model.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()

acc = 100 * correct / len(test_data)
print(f"\nBaseline Results:")
print(f"  Test Accuracy : {acc:.2f}%")
print(f"  Training Time : {elapsed:.2f}s")

# quick sanity check - show one prediction
sample_img, sample_label = test_data[0]
with torch.no_grad():
    pred = model(sample_img.unsqueeze(0)).argmax().item()
print(f"\nSample prediction: model says {pred}, actual label is {sample_label}")

torch.save(model.state_dict(), 'baseline_model.pth')