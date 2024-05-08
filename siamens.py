import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SiameseUNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseUNet, self).__init__()
        
        # Load pre-trained ResNet34 as encoder
        self.encoder = models.resnet34(pretrained=pretrained)
        
        # Remove fully connected layers from ResNet34
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Define decoder layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, img1, img2):
        # Encode both images
        features1 = self.encoder(img1)
        features2 = self.encoder(img2)
        
        # Concatenate features
        x = torch.cat((features1, features2), dim=1)
        
        # Decoder
        x = F.relu(self.upconv1(x))
        x = torch.cat((features1, x), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.upconv2(x))
        x = torch.cat((features2, x), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # Output layer
        x = self.conv7(x)
        return x
