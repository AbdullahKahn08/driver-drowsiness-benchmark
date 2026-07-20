import torch.nn
from torchvision.models import googlenet

def get_googlenet():
    model = googlenet(weights="IMAGENET1K_V1")
    
    for param in model.conv1.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(1024,2)

    return model

