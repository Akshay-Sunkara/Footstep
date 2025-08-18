import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, root_dirs, labels): 
        self.samples = []
        for root_dir, label in zip(root_dirs, labels):
            files = glob.glob(os.path.join(root_dir, "*.npy"))
            for f in files:
                self.samples.append((f, label))

    def __len__(self):  
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)  
        mel_tensor = torch.tensor(mel, dtype=torch.float32)
        return mel_tensor, torch.tensor(label, dtype=torch.long)

gunshotsE_dir = "Training/GunshotsEMEL"
gunshotsW_dir = "Training/GunshotsWMEL"
noGunshots_dir = "Training/NoGunshotsMEL"

dataset = NpyDataset(
    root_dirs=[gunshotsE_dir, gunshotsW_dir, noGunshots_dir],
    labels=[0, 1, 2]
)

print(f"Total samples: {len(dataset)}")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

class_names = ["Gunshot East", "Gunshot West", "No Gunshots"]
print(f"Classes: {class_names}")

print("Calculating normalization statistics...")
all_data = []
for i, (data, _) in enumerate(train_loader):
    all_data.append(data)
    if i % 10 == 0:
        print(f"Processing batch {i}/{len(train_loader)}")

all_data = torch.cat(all_data, dim=0)
global_mean = all_data.mean()
global_std = all_data.std()

print(f"Global mean: {global_mean:.4f}, Global std: {global_std:.4f}")

torch.save({'mean': global_mean, 'std': global_std}, 'normalization_stats.pth')
print("Saved normalization statistics")

class NpyDatasetNormalized(Dataset):
    def __init__(self, root_dirs, labels, global_mean, global_std):
        self.samples = []
        self.global_mean = global_mean
        self.global_std = global_std
        for root_dir, label in zip(root_dirs, labels):
            files = glob.glob(os.path.join(root_dir, "*.npy"))
            for f in files:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)  
        mel_tensor = torch.tensor(mel, dtype=torch.float32)

        mel_tensor = (mel_tensor - self.global_mean) / (self.global_std + 1e-9)
        return mel_tensor, torch.tensor(label, dtype=torch.long)

dataset_normalized = NpyDatasetNormalized(
    root_dirs=[gunshotsE_dir, gunshotsW_dir, noGunshots_dir],
    labels=[0, 1, 2],
    global_mean=global_mean,
    global_std=global_std
)

train_dataset, test_dataset = torch.utils.data.random_split(dataset_normalized, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 40, 1200) 
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, 100)
        self.fc4 = nn.Linear(100, 3) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0007)

print("Starting training...")

for epoch in range(10):
    running_loss = 0.0
    net.train()
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 2 == 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), "gunshot_model.pth")
print("Model saved as gunshot_model.pth")

net.eval()
correct = 0
total = 0
test_loss = 0.0
class_correct = [0, 0, 0]
class_total = [0, 0, 0]

with torch.no_grad():  
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

print(f"\nTest Results:")
print(f"Test Loss: {test_loss/len(test_loader):.3f}")
print(f"Overall Accuracy: {100 * correct / total:.2f}%")

print(f"\nPer-class Accuracy:")
for i in range(3):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"{class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{class_names[i]}: No test samples")

print("\nFiles saved:")
print("- gunshot_model.pth (trained model)")
print("- normalization_stats.pth (normalization parameters)")