import torch.nn
from torchvision.models import mobilenet_v3_large

class EarlyFusionMobileNetv3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        eye_backbone = mobilenet_v3_large(weights="IMAGENET1K_V2")
        mouth_backbone = mobilenet_v3_large(weights="IMAGENET1K_V2")

        self.eye_branch = eye_backbone.features[0:9]
        self.mouth_branch = mouth_backbone.features[0:9]

        for param in self.eye_branch[0].parameters():
            param.requires_grad = False
        
        for param in self.mouth_branch[0].parameters():
            param.requires_grad = False

        self.fusion_conv = torch.nn.Conv2d(160,80,kernel_size=1)

        self.shared_layer = eye_backbone.features[9:17]

        self.avgpool = eye_backbone.avgpool

        self.fc = torch.nn.Linear(960,2)

    def forward(self,eyes_input,mouth_input):
        eye_features = self.eye_branch(eyes_input)
        mouth_features = self.mouth_branch(mouth_input)
        features_concatenate = torch.cat([eye_features,mouth_features],dim=1)
        early_fusion = self.fusion_conv(features_concatenate)
        combined_features = self.shared_layer(early_fusion)
        combined_features = self.avgpool(combined_features)
        combined_features = torch.flatten(combined_features,1)
        output = self.fc(combined_features)
        return output



