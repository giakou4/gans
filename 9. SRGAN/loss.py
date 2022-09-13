import torch.nn as nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.loss(x_features, y_features)


