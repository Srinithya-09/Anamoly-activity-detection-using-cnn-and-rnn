import os
import torch
from torch.utils.data import Dataset

from utils import load_video_frames


class VideoDataset(Dataset):
    """
    Loads videos from a directory structured as:

        root_dir/
        ├── normal/
        │   ├── video1.mp4
        │   └── ...
        └── anomaly/
            ├── video1.mp4
            └── ...

    Each video is converted into a fixed-length sequence of frames
    via load_video_frames(), and labeled 0 (normal) or 1 (anomaly).
    """

    LABEL_MAP = {"normal": 0, "anomaly": 1}

    def __init__(self, root_dir, num_frames=16, frame_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.samples = self._index_videos()

    def _index_videos(self):
        samples = []
        for class_name, label in self.LABEL_MAP.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    samples.append((os.path.join(class_dir, filename), label))

        if not samples:
            raise RuntimeError(
                f"No videos found under '{self.root_dir}'. "
                f"Expected subfolders: {list(self.LABEL_MAP.keys())}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = load_video_frames(
            video_path,
            num_frames=self.num_frames,
            frame_size=self.frame_size,
        )  # shape: (num_frames, H, W, C)

        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (num_frames, C, H, W)

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        label = torch.tensor(label, dtype=torch.float32)
        return frames, label


if __name__ == "__main__":
    # Quick sanity check
    dataset = VideoDataset(root_dir="dataset")
    print(f"Found {len(dataset)} videos")
    sample_frames, sample_label = dataset[0]
    print(f"Sample frames shape: {sample_frames.shape}, label: {sample_label.item()}")
