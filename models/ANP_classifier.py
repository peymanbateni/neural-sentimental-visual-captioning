import torch
import torch.nn as nn
import torchvision
from torch import optim

class ANPClassifier(nn.Module):
    def __init__(self, output_size):
        super(ANPClassifier, self).__init__()
        
        self.output_size = output_size
        self.resnet = torchvision.models.resnet101(pretrained=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=output_size, bias=True)

    def forward(self, X):
        return self.resnet(X)

print(ANPClassifier(1200))