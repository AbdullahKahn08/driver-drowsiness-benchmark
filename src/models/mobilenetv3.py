import torch.nn
from torchvision.models import mobilenet_v3_large

def get_mobilenetv3():
    model = mobilenet_v3_large(weights = "IMAGENET1K_V2")

    model.classifier[-1] = torch.nn.Linear(1280,2)

    return model