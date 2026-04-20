import torch
import torch.nn as nn
import torchvision.models as models

class VGGExtractor(nn.Module):
    def __init__(self):
        super(VGGExtractor, self).__init__()
        # Load VGG19 pre-trained on ImageNet
        vgg19 = models.vgg19(pretrained=True)
        
        # We only need the first few layers (up to relu5_4) to detect textures
        self.features = nn.Sequential(*list(vgg19.features.children())[:35])
        
        # Freeze VGG (we don't want to train it, just use it as a judge)
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.features(x)