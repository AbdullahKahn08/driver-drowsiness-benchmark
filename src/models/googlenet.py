import torch.nn
from torchvision.models import googlenet

def get_googlenet():
    model = googlenet(weights="IMAGENET1K_V1")

    model.fc = torch.nn.Linear(1024,2)

    return model

