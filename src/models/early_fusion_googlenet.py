import torch.nn
from torchvision.models import googlenet

class EarlyFusionGoogleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        eye_backbone = googlenet(weights="IMAGENET1K_V1")
        mouth_backbone = googlenet(weights="IMAGENET1K_V1")

        self.eye_branch = torch.nn.Sequential(
            eye_backbone.conv1,
            eye_backbone.maxpool1,
            eye_backbone.conv2,
            eye_backbone.conv3,
            eye_backbone.maxpool2,
            eye_backbone.inception3a,
            eye_backbone.inception3b
        )

        self.mouth_branch = torch.nn.Sequential(
            mouth_backbone.conv1,
            mouth_backbone.maxpool1,
            mouth_backbone.conv2,
            mouth_backbone.conv3,
            mouth_backbone.maxpool2,
            mouth_backbone.inception3a,
            mouth_backbone.inception3b
        )

        self.fusion_conv = torch.nn.Conv2d(960,480,kernel_size=1)

        self.shared_layer = torch.nn.Sequential(
            eye_backbone.maxpool3,
            eye_backbone.inception4a,
            eye_backbone.inception4b,
            eye_backbone.inception4c,
            eye_backbone.inception4d,
            eye_backbone.inception4e,
            eye_backbone.maxpool4,
            eye_backbone.inception5a,
            eye_backbone.inception5b,
            eye_backbone.avgpool,
            eye_backbone.dropout
        )

        self.fc = torch.nn.Linear(1024,2)

    def forward(self,eye_input,mouth_input):
        eye_features = self.eye_branch(eye_input)
        mouth_features = self.mouth_branch(mouth_input)
        features_concatenate = torch.cat([eye_features,mouth_features],dim=1)
        early_fusion = self.fusion_conv(features_concatenate)
        combined_features = self.shared_layer(early_fusion)
        combined_features = torch.flatten(combined_features,1)
        output = self.fc(combined_features)
        return output
