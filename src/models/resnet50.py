import torch.nn
from torchvision.models import resnet50

def get_resnet50():
    model = resnet50(weights = "IMAGENET1K_V2")
    
    for paramter in model.parameters():
        paramter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    
    model.fc = torch.nn.Linear(2048,2)

    return model