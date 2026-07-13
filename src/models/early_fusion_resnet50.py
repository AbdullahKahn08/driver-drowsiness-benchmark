import torch.nn
from torchvision.models import resnet50

class EarlyFusionResNet(torch.nn.Module):
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
            eye_backbone.layer2
        )

        self.mouth_branch = torch.nn.Sequential(
            mouth_backbone.conv1,
            mouth_backbone.bn1,
            mouth_backbone.relu,
            mouth_backbone.maxpool,
            mouth_backbone.layer1,
            mouth_backbone.layer2
        )

        for param in self.eye_branch.parameters():
            param.requires_grad = False

        for param in self.mouth_branch.parameters():
            param.requires_grad = False

        self.fusion_conv = torch.nn.Conv2d(1024,512,kernel_size=1)

        self.shared_layer = torch.nn.Sequential(
            eye_backbone.layer3,
            eye_backbone.layer4,
            eye_backbone.avgpool
        )

        self.fc = torch.nn.Linear(2048,2)
    
    def forward(self,eyes_input,mouths_input):
        eye_features = self.eye_branch(eyes_input)
        mouth_features = self.mouth_branch(mouths_input)
        features_concatente = torch.cat([eye_features,mouth_features],dim=1)
        early_fusion = self.fusion_conv(features_concatente)
        combined_features = self.shared_layer(early_fusion)
        combined_features = torch.flatten(combined_features,1)
        output = self.fc(combined_features)
        return output
