import torch
import torchvision

class MLC(torch.nn.Module):
    @property
    def transforms(self):
        return torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms
    
    def __init__(self, num_classes, pretrained_mlp=None):
        super().__init__()

        self.resnet50 = torch.nn.Sequential(
            *list(
                torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                ).children()
            )[:-1]
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes)
        )
        
        if pretrained_mlp is not None:
            self.classifier.load_state_dict(pretrained_mlp, map_location='cpu')

        for layer in self.resnet50.parameters():
            layer.requires_grad = False

    def forward(self, inputs):
        features = self.resnet50(inputs).flatten(1)
        return self.classifier(features)

if __name__ == '__main__':
    model = MLC(10)
    print(model)
    
    import numpy as np
    inputs = torch.from_numpy(np.random.rand(10, 3, 224, 224).astype(np.float32))
    
    outputs = model(inputs) 
    print(outputs.shape)