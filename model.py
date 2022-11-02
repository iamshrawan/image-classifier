import torch.nn as nn
from torchvision import models

class Classifier(nn.Module):
    def __init__(self, in_channel=3, pre_trained=False, num_classes=10):
        super(Classifier, self).__init__()
        self.in_channel = in_channel
        self.pre_trained = pre_trained
        self.num_classes = num_classes
        self.build_model()
        
        
    def build_model(self):
        print(f'Building resnet_34 model (pretrained={self.pre_trained})!!')
        model_resnet = models.resnet34(pretrained=self.pre_trained)
        num_ftrs = model_resnet.fc.in_features
        model_resnet.fc = nn.Linear(num_ftrs, self.num_classes)
        self.features = nn.Sequential(*list(model_resnet.children())[:-1])
        self.classifier = model_resnet.fc
        if self.pre_trained:
                for param in self.features[0:7].parameters():
                    param.requires_grad = False 

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.shape[0], -1)
        return self.classifier(feat)
        