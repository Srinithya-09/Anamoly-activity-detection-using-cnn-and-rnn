import cv2
import os
import torch
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

def load_video_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame)
        frames.append(frame_tensor)
    cap.release()
    if len(frames) < max_frames:
        frames += [frames[-1]] * (max_frames - len(frames))  # pad
    return torch.stack(frames)  # (seq_len, C, H, W)
