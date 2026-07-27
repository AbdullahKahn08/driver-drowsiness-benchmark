import torch.nn
from torchvision.models import googlenet

class LateFusionGoogleNet(torch.nn.Module):
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
            eye_backbone.inception3b,
            eye_backbone.maxpool3,
            eye_backbone.inception4a,
            eye_backbone.inception4b,
            eye_backbone.inception4c,
            eye_backbone.inception4d,
            eye_backbone.inception4e,
            eye_backbone.maxpool4,
            eye_backbone.inception5a,
            eye_backbone.inception5b
        )

        self.mouth_branch = torch.nn.Sequential(
            mouth_backbone.conv1,
            mouth_backbone.maxpool1,
            mouth_backbone.conv2,
            mouth_backbone.conv3,
            mouth_backbone.maxpool2,
            mouth_backbone.inception3a,
            mouth_backbone.inception3b,
            mouth_backbone.maxpool3,
            mouth_backbone.inception4a,
            mouth_backbone.inception4b,
            mouth_backbone.inception4c,
            mouth_backbone.inception4d,
            mouth_backbone.inception4e,
            mouth_backbone.maxpool4,
            mouth_backbone.inception5a,
            mouth_backbone.inception5b
        )

        self.avg_pool = eye_backbone.avgpool

        self.fc = torch.nn.Linear(2048,2)

    def forward(self,eyes_input,mouth_input):
        eye_features = self.eye_branch(eyes_input)
        mouth_features = self.mouth_branch(mouth_input)
        eye_avg_pool = self.avg_pool(eye_features)
        mouth_avg_pool = self.avg_pool(mouth_features)
        combined_features = torch.cat([eye_avg_pool,mouth_avg_pool],dim=1)
        combined_features = torch.flatten(combined_features,1)
        output = self.fc(combined_features)
        return output
