import torch

class MultiLabelClassification(torch.nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feature_extractor.output_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes)
        )

        for layer in self.feature_extractor.parameters():
            layer.requires_grad = False

    def forward(self, imgs):

        features = self.feature_extractor(
            imgs.to(dtype = self.feature_extractor.conv1.weight.dtype)
        ).to(dtype = self.classifier[0].weight.dtype)

        return self.classifier(features)