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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
try:
    net.load_state_dict(torch.load("Overlay/gunshot_model.pth", map_location='cpu'))
    print("Model loaded successfully")
except FileNotFoundError:
    print("ERROR: gunshot_model.pth not found. Please run training script first.")
    sys.exit(1)

net.eval()

try:
    norm_stats = torch.load('Overlay/normalization_stats.pth', map_location='cpu')
    global_mean = norm_stats['mean']
    global_std = norm_stats['std']
    print(f"Loaded normalization stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}")
except FileNotFoundError:
    print("ERROR: normalization_stats.pth not found. Please run training script first.")
    sys.exit(1)

class_names = ["Gunshot East", "Gunshot West", "No gunshots"]

os.makedirs("Test", exist_ok=True)

print("Starting real-time gunshot detection...")
print("Press Ctrl+C to stop")
print(f"Classes: {class_names}")
print("-" * 50)

try:
    detection_count = 0
    while True:
        detection_count += 1
        
        duration = 2
        fs = 44100
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        
        write("Test/output.wav", fs, myrecording)
        
        time.sleep(0.1)
        
        try:
            y_stereo, sr = librosa.load("Test/output.wav", sr=None, mono=False)
            y_left = y_stereo[0]
            y_right = y_stereo[1]
           
            left_mel = librosa.feature.melspectrogram(y=y_left, sr=sr)
            left_mel_db = librosa.power_to_db(left_mel, ref=np.max)
            right_mel = librosa.feature.melspectrogram(y=y_right, sr=sr)
            right_mel_db = librosa.power_to_db(right_mel, ref=np.max)
            
            stereo_mel_db = np.stack([left_mel_db, right_mel_db], axis=0)
            stereo_mel_tensor = torch.tensor(stereo_mel_db, dtype=torch.float32)
            stereo_mel_tensor = stereo_mel_tensor.unsqueeze(0)
            
            stereo_mel_tensor = (stereo_mel_tensor - global_mean) / (global_std + 1e-9)
            
            with torch.no_grad():
                output = net(stereo_mel_tensor)
                probabilities = F.softmax(output, dim=1)
                max_prob, prediction = torch.max(probabilities, dim=1)
                
                predicted_class = prediction.item()
                prediction = predicted_class
                confidence = max_prob.item()
                print(predicted_class)
                
        except Exception as e:
            print(f"Error processing audio: {e}")
        
        sys.stdout.flush()
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopping real-time detection...")
    print("Goodbye!")
except Exception as e:
    print(f"Unexpected error: {e}")