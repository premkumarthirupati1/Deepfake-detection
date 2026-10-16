"""
MesoInception-4 PyTorch Architecture
Based on the seminal deepfake forensics research:
'MesoNet: a Compact Facial Video Forgery Detection Network' by Darius Afchar et al.
Directly referenced in IEEE Access: 'Deepfake Detection: A Systematic Literature Review'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MesoInceptionBlock(nn.Module):
    """MesoInception module with multi-scale receptive fields (1x1, 3x3, 5x5)."""
    def __init__(self, in_channels, a, b, c, d):
        super(MesoInceptionBlock, self).__init__()
        # Branch 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, a, kernel_size=1, padding=0),
            nn.BatchNorm2d(a),
            nn.LeakyReLU(0.1)
        )
        # Branch 2: 1x1 conv -> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=1, padding=0),
            nn.BatchNorm2d(b),
            nn.LeakyReLU(0.1),
            nn.Conv2d(b, b, kernel_size=3, padding=1),
            nn.BatchNorm2d(b),
            nn.LeakyReLU(0.1)
        )
        # Branch 3: 1x1 conv -> 5x5 (as 3x3 dilated) conv
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=1, padding=0),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.1),
            nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.1)
        )
        # Branch 4: MaxPool -> 1x1 conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, d, kernel_size=1, padding=0),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class MesoInception4(nn.Module):
    """
    MesoInception-4 deep convolutional forensic network.
    Evaluates micro-textures, compression artifacts, and blending boundaries in face crops.
    """
    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()
        
        # Inception Layer 1
        self.inc1 = MesoInceptionBlock(3, 1, 4, 4, 2)  # output: 11 channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Inception Layer 2
        self.inc2 = MesoInceptionBlock(11, 2, 4, 4, 2) # output: 12 channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Standard Conv Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Standard Conv Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        
        # Adaptive pooling to support any input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 8 * 8, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.inc1(x))
        x = self.pool2(self.inc2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        out = self.classifier(x)
        return out
