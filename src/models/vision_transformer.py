import torch
from torchvision.models import vit_b_16

def vision_transformer():
    model = vit_b_16(weights="IMAGENET1K_V1")

    model.heads.head = torch.nn.Linear(768,2)

    return model


