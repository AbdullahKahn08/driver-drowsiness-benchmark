import torch.nn
from torchvision.models import resnet50

class LateFusionResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        eye_backbone = resnet50(weights="IMAGENET1K_V2")
        mouth_backbone = resnet50(weights="IMAGENET1K_V2")
        
        self.eye_branch = torch.nn.Sequential(
            eye_backbone.conv1,
            eye_backbone.bn1,
            eye_backbone.relu,
            eye_backbone.maxpool,
            eye_backbone.layer1,
            eye_backbone.layer2,
            eye_backbone.layer3,
            eye_backbone.layer4,
            eye_backbone.avgpool
        )

        self.mouth_branch = torch.nn.Sequential(
            mouth_backbone.conv1,
            mouth_backbone.bn1,
            mouth_backbone.relu,
            mouth_backbone.maxpool,
            mouth_backbone.layer1,
            mouth_backbone.layer2,
            mouth_backbone.layer3,
            mouth_backbone.layer4,
            mouth_backbone.avgpool
        )

        #self.fusion_conv = torch.nn.Conv2d(4096,2048,kernel_size=1)

        self.fc = torch.nn.Linear(4096,2)
    
    def forward(self,eyes_input,mouths_input):
        eye_features = self.eye_branch(eyes_input)
        mouth_features = self.mouth_branch(mouths_input)
        features_concatente = torch.cat([eye_features,mouth_features],dim=1)
        #late_fusion = self.fusion_conv(features_concatente)
        combined_features = torch.flatten(features_concatente,1)
        output = self.fc(combined_features)
        return output
