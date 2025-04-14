import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM_Model(nn.Module):
    def __init__(self, cnn_out_dim=512, hidden_dim=256, num_layers=1):
        super(CNN_LSTM_Model, self).__init__()
        
        # Pretrained CNN (e.g., ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification: normal vs anomaly

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        cnn_features = []

        for t in range(seq_len):
            out = self.cnn(x[:, t])           # (batch, 512, 1, 1)
            out = out.view(batch_size, -1)    # (batch, 512)
            cnn_features.append(out)
        
        cnn_seq = torch.stack(cnn_features, dim=1)  # (batch, seq_len, 512)
        lstm_out, _ = self.lstm(cnn_seq)
        final_out = self.fc(lstm_out[:, -1, :])     # Use last LSTM output
        return torch.sigmoid(final_out)
