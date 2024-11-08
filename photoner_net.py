import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from unet_model import UNet

# Define the CNN model (U-Net like architecture)
class PhotonerNet(UNet):
    def __init__(self):
        super(PhotonerNet, self).__init__(n_channels=1, n_classes=1, bilinear=False)

        self.pack = transforms.Compose([
            # lambda x: torch.log10(x) + 2
            # lambda x: pow(x, 0.1),
            # transforms.Normalize(0.5, 0.5)
        ])

        self.unpack = transforms.Compose([
            # lambda x: torch.pow(10, x - 2)
            # lambda x: pow(x, 10)
            # transforms.Normalize(-1, 2),
        ])


    def forward(self, x):
        o = torch.zeros_like(x)

        x = self.pack(x)

        for c in range(0, o.size(1)):
            input_channel = x[:, c, :, :].unsqueeze(1)
            output_channel = super(PhotonerNet, self).forward(input_channel)
            o[:, c, :, :] = output_channel[:, 0, :, :]
        
        o = self.unpack(o)

        return o