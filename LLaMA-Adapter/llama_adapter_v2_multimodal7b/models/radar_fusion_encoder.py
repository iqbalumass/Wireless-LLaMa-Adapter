# models/radar_fusion_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(self.fc(x), dim=-1)
        return x
