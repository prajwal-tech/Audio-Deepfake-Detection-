import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import roc_curve, auc
import torchaudio.transforms as transforms
from rawnet2 import RawNet2  

REAL_AUDIO_PATH = "path/to/ASVspoof2019_LA/train/bonafide"
FAKE_AUDIO_PATH = "path/to/ASVspoof2019_LA/train/fake"


TARGET_SAMPLE_RATE = 16000  
MAX_AUDIO_LENGTH = TARGET_SAMPLE_RATE * 4 


class ASVspoofDataset(Dataset):
    def __init__(self, real_path, fake_path):
        self.real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('.wav')]
        self.fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith('.wav')]
        self.files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        self.resampler = transforms.Resample(orig_freq=48000, new_freq=TARGET_SAMPLE_RATE)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        
        waveform, sample_rate = torchaudio.load(file_path)

        
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = self.resampler(waveform)

        
        waveform = waveform / torch.max(torch.abs(waveform))
        if waveform.shape[1] > MAX_AUDIO_LENGTH:
            waveform = waveform[:, :MAX_AUDIO_LENGTH]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, MAX_AUDIO_LENGTH - waveform.shape[1]))

        return waveform, torch.tensor(label, dtype=torch.long)


dataset = ASVspoofDataset(REAL_AUDIO_PATH, FAKE_AUDIO_PATH)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RawNet2(num_classes=2).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)  # Learning rate decay


num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


torch.save(model.state_dict(), "rawnet2_deepfake_detector.pth")


model.eval()
all_labels = []
all_scores = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  

        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
eer_threshold = thresholds[np.nanargmin(np.abs((1 - tpr) - fpr))]
eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
auc_score = auc(fpr, tpr)

print(f"Model EER: {eer:.2%}")
print(f"Model AUC: {auc_score:.4f}")


with open("evaluation_results.txt", "w") as f:
    f.write(f"Equal Error Rate (EER): {eer:.4f}\n")
    f.write(f"Area Under Curve (AUC): {auc_score:.4f}\n")


model.load_state_dict(torch.load("rawnet2_deepfake_detector.pth"))
model.eval()

print("Model successfully trained and saved!")