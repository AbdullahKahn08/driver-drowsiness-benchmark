import torch.nn
from torchvision.models import mobilenet_v3_large

def get_mobilenetv3():
    model = mobilenet_v3_large(weights = "IMAGENET1K_V2")
    
    for paramter in model.features[0].parameters():
        paramter.requires_grad = False

    model.classifier[-1] = torch.nn.Linear(1280,2)

    return model