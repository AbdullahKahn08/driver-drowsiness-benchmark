import torch.nn
from torchvision.models import mobilenet_v3_large

class LateFusionMobileNetv3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,eyes_input,mouth_input):
        pass