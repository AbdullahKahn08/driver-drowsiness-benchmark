import torch.nn
from torchvision.models import resnet50

def get_resnet50():
    model = resnet50(weights = "IMAGENET1K_V2")
    
    model.fc = torch.nn.Linear(2048,2)

    return model