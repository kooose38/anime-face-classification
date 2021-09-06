import torch.nn as nn 
import torch, torchvision 
from torchvision.models import vgg19 

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.vgg = vgg19(pretrained=True)
        for w in self.vgg.parameters():
            w.requires_grad = False 
        self.fc = nn.Linear(1000, n_classes)

    def forward(self, x):
        y = self.vgg(x)
        y = self.fc(y)
        return y 