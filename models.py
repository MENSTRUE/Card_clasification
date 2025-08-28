import torch.nn as nn
from torchvision import models


def create_vgg_model(num_classes: int):
    model = models.vgg16(weights='IMAGENET1K_V1')

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, num_classes)
    )
    return model