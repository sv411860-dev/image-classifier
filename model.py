import torch
import torch.nn as nn
from torchvision import models

class CatDogCNN(nn.Module):
    """
    Custom Convolutional Neural Network built from scratch for binary image classification.
    """
    def __init__(self, num_classes=2):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if x.device.type == "mps":
            x = self.adaptive_pool(x.cpu()).to("mps")
        else:
            x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class CatDogTransfer(nn.Module):
    """
    High-accuracy classifier using MobileNetV3 pretrained on ImageNet (1.4M images),
    fine-tuned on Cats vs Dogs.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CatDogTransfer, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model(model_type="transfer", num_classes=2, pretrained=True):
    if model_type == "transfer":
        return CatDogTransfer(num_classes=num_classes, pretrained=pretrained)
    else:
        return CatDogCNN(num_classes=num_classes)

if __name__ == "__main__":
    import os
    import runpy
    app_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    if os.path.exists(app_file):
        runpy.run_path(app_file, run_name="__main__")

