import os
import torch
from torch.utils.data import Dataset, DataLoader
from model import CNN_LSTM_Model
from utils import load_video_frames
import glob

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, class_name in enumerate(['normal', 'anomaly']):
            video_paths = glob.glob(os.path.join(root_dir, class_name, '*.mp4'))
            for path in video_paths:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = load_video_frames(video_path)  # (seq_len, C, H, W)
        return frames, torch.tensor([label], dtype=torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN_LSTM_Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

dataset = VideoDataset('dataset')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in dataloader:
        videos, labels = batch
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
