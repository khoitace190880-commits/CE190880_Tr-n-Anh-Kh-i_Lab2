import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 7

class EmotionCNN(nn.Module):
    """Custom CNN checkpoint-compatible with the trained FER-2013 M1 model."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Four VGG-style blocks; each block has two Conv-BN-ReLU layers and pooling.
        # Keep nested Sequential indices stable because they match cnn_best.pth.
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EmotionResNet18(nn.Module):
    """Transfer-learning model; input is RGB 224x224."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        # Keep the new classification head trainable even when freezing the backbone.
        for p in self.backbone.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


def build_model(name, pretrained=True, freeze_backbone=False):
    name = name.lower()
    if name in ('cnn', 'm1'):
        return EmotionCNN()
    if name in ('transfer', 'resnet18', 'm2'):
        return EmotionResNet18(pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model: {name}. Use cnn or transfer.")
