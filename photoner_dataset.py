from torch.utils.data import Dataset
import os.path
import OpenEXR
from torch import tensor
import torchvision.io
import torch

class PhotonerDataset(Dataset):
    def __init__(self, folder, profileIndex, tonemapped, transform=None):
        self.folder = folder
        self.profileIndex = profileIndex
        self.tonemapped = tonemapped
        self.transform = transform

    def __len__(self):
        if(self.tonemapped):
            ext = "png"
        else:
            ext = "exr"
        input_i = 0
        train_i = 0
        while(os.path.isfile(os.path.join(self.folder, f"Input_{self.profileIndex}_{input_i:04d}.{ext}"))):
            input_i += 1
        while(os.path.isfile(os.path.join(self.folder, f"Output_{train_i:04d}.{ext}"))):
            train_i += 1
        return min(input_i, train_i)
    
    def __getitem__(self,idx):
        if(self.tonemapped):
            ext = "png"
        else:
            ext = "exr"
        input_path = os.path.join(self.folder, f"Input_{self.profileIndex}_{idx:04d}.{ext}")
        train_path = os.path.join(self.folder, f"Output_{idx:04d}.{ext}")

        if(not self.tonemapped):
            with OpenEXR.File(input_path) as exrFile:
                RGB = exrFile.channels()["RGBA"].pixels
                input_image = tensor(RGB, dtype=torch.float32).permute(2,1,0)[0:3, :, :]
            with OpenEXR.File(train_path) as exrFile:
                RGB = exrFile.channels()["RGBA"].pixels
                train_image = tensor(RGB, dtype=torch.float32).permute(2,1,0)[0:3, :, :]

        else:
            input_image = torchvision.io.decode_image(input_path, mode="RGB").type(torch.float32)
            train_image = torchvision.io.decode_image(train_path, mode="RGB").type(torch.float32)

        if(self.transform):
            input_image = self.transform(input_image)
            train_image = self.transform(train_image)

        item = (input_image, train_image)

        return item
                # height, width = RGB.shape[0:2]
                # for y in range(height):
                #     for x in range(width):
                #         pixel = (RGB[y, x, 0], RGB[y, x, 1], RGB[y, x, 2])
                #         print(f"pixel[{y}][{x}]={pixel}")
