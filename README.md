# Anomaly Activity Detection using CNN and RNN

A hybrid deep learning pipeline that detects abnormal human activity in video by combining a **CNN (ResNet-18)** for spatial feature extraction with an **LSTM** for modeling temporal dependencies across frames.

Built for use cases like video surveillance, healthcare monitoring, and industrial automation, where catching irregular or unexpected behavior matters.

## Overview

Each input video is sampled into a fixed-length sequence of frames. A pretrained ResNet-18 extracts a spatial feature vector from every frame, and these per-frame features are fed into an LSTM that learns how the scene evolves over time. The final LSTM hidden state is passed through a fully connected layer with a sigmoid activation, producing a binary prediction: **normal** vs. **anomaly**.

```
Video → Frame Sampling → CNN (ResNet-18) → Per-frame Features → LSTM → FC + Sigmoid → Normal / Anomaly
```

## Project Structure

```
.
├── model.py    # CNN_LSTM_Model: ResNet-18 backbone + LSTM + classification head
├── train.py    # VideoDataset loader and training loop
├── utils.py    # Video frame extraction and preprocessing
└── README.md
```

## How It Works

- **`utils.py`** — `load_video_frames()` reads a video with OpenCV, converts frames to RGB, resizes them to 224×224, and pads/truncates the sequence to a fixed number of frames (default: 16).
- **`model.py`** — `CNN_LSTM_Model` runs each frame through a pretrained ResNet-18 (final FC layer removed) to get a 512-dim feature vector, stacks these into a sequence, and passes it through an LSTM. The last LSTM output feeds a linear layer + sigmoid for binary classification.
- **`train.py`** — `VideoDataset` expects videos organized into `normal/` and `anomaly/` subfolders, and trains the model using binary cross-entropy loss with the Adam optimizer.

## Dataset Format

Organize your videos as follows before training:

```
dataset/
├── normal/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── anomaly/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

## Setup

```bash
git clone https://github.com/Srinithya-09/Anamoly-activity-detection-using-cnn-and-rnn.git
cd Anamoly-activity-detection-using-cnn-and-rnn
pip install torch torchvision opencv-python numpy
```

## Training

Place your dataset in a `dataset/` folder as described above, then run:

```bash
python train.py
```

This will train `CNN_LSTM_Model` for 5 epochs, printing the average loss per epoch. Training automatically uses a GPU if one is available.

## Model Details

| Component | Detail |
|---|---|
| CNN backbone | ResNet-18 (ImageNet-pretrained, final FC removed) |
| Sequence model | 1-layer LSTM, hidden size 256 |
| Input | Sequence of 16 RGB frames, resized to 224×224 |
| Output | Single sigmoid probability (anomaly likelihood) |
| Loss | Binary Cross-Entropy (BCELoss) |
| Optimizer | Adam, lr=1e-4 |

## Possible Extensions

- Add a validation split and accuracy/F1/AUC metrics
- Swap the LSTM for a GRU or a Transformer-based temporal encoder
- Handle class imbalance with weighted loss or oversampling
- Add checkpointing and inference scripts for real-time video streams
- Log training runs with TensorBoard or Weights & Biases

## Tech Stack

- Python
- PyTorch / Torchvision
- OpenCV

## License

No license specified yet — consider adding one (e.g., MIT) if you'd like others to reuse this code.
