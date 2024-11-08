import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.onnx
import torchvision.io as io
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL.Image as Image
import os
from photoner_net import PhotonerNet
from unet_model import UNet
from matplotlib import pyplot as plt
from photoner_dataset import PhotonerDataset
import shutil
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights

# TODO:
# - Add Test Phase
#
#

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(pred,actual) # torch.sqrt(self.mse(torch.sqrt(torch.clamp(pred, min=0)), torch.sqrt(torch.clamp(actual, min=0))))
    
def save_model(model, whatever, path):
    model.eval()
    # input_tensor = torch.rand((1, 1, 512, 512), dtype=torch.float32)
    torch.onnx.export(model,               # model being run
                  whatever,                         # model input (or a tuple for multiple inputs)
                  path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  optimize=True,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {1 : 'channels', 2 : 'width', 3 : 'height'},    # variable length axes
                                'output' : {1 : 'channels', 2 : 'width', 3 : 'height'}})
    
def display_sample(input, output, expect):
    f = plt.figure()
    ax1 = f.add_subplot(1,3, 1)
    ax1.set_title('Input')
    plt.imshow(  input.cpu().permute(1, 2, 0)  )
    ax2 = f.add_subplot(1,3, 2)
    ax2.set_title('Actual')
    plt.imshow(  output.cpu().permute(1, 2, 0)  )
    ax3 = f.add_subplot(1,3, 3)
    ax3.set_title('Expected')
    plt.imshow(  expect.cpu().permute(1, 2, 0)  )
    plt.show(block=True)

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
data_folder = '/home/evan/Pictures/training_output/2024-10-29-22-21-11'
input_image_folder = '~/Pictures/photoner_input'
train_image_folder = '~/Pictures/photoner_train'
num_epochs = 10
batch_size = 64
learning_rate = 0.001
training_data_ratio = 1.00

# Load the datasets
transform = transforms.Compose([
   # transforms.ToTensor()
   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

photoner_dataset = PhotonerDataset(data_folder, profileIndex=2, tonemapped=False, transform=transform)
photoner_loader = data.DataLoader(photoner_dataset, batch_size=batch_size, shuffle=False)


# input_dataset = datasets.ImageFolder(root=input_image_folder, transform=transform)
# input_loader = data.DataLoader(input_dataset, batch_size=batch_size, shuffle=False)

# train_dataset = datasets.ImageFolder(root=train_image_folder, transform=transform)
# train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
# model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
model = PhotonerNet().to(device)
vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).to(device)
mse = nn.MSELoss().to(device)
msle = RMSLELoss().to(device)
mae = nn.L1Loss()
cel = nn.CrossEntropyLoss().to(device)

def vgg_criterion(output, train):
    # output_image = normalize_batch(output_image)
    # train_image = normalize_batch(train_image)
    output_features = vgg(output)
    train_features = vgg(train)
    return mse(output_features, train_features)

    
criterion = msle
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
image_count = len(photoner_dataset)
training_image_count = int(image_count * training_data_ratio) 
loss_trend = -1
for epoch in range(num_epochs):
    for i in range(0, training_image_count):
        (input_image, train_image) = photoner_dataset[i]
        # (input_image, _) = input_dataset[i]
        # (train_image, _) = train_dataset[i]

        input_image = input_image.to(device).unsqueeze(0)
        train_image = train_image.to(device).unsqueeze(0)
        output_image = torch.zeros(1, 3, 512, 512).to(device)

        for c in range(0, 3):
            input_channel = input_image[:, c, :, :].unsqueeze(1)
            train_channel = train_image[:, c, :, :].unsqueeze(1)
        
            # Forward pass
            output_channel = model(input_channel)
            output_image[:, c, :, :] = output_channel[:, 0, :, :]
        loss = criterion(output_image, train_image)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(loss_trend == -1):
            loss_trend = loss.item()
        else:
            loss_trend = max(loss.item(), loss_trend * 0.9 + loss.item() * 0.1)

        if (i+1) % 100 == 0:
            save_model(model, input_image, 'Assets/Resources/checkpoint.onnx')
            model.train()
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{training_image_count}], Loss: {loss_trend:.8f}')
            # display_sample(input_image_cpu, output_image.squeeze().detach(), train_image_cpu)
            # display_sample(input_image_cpu *10, output_image.squeeze().detach()*10, train_image_cpu*10)


if(os.path.exists('Assets/Resources/model.onnx')):
    shutil.copy('Assets/Resources/model.onnx', 'Assets/Resources/model_backup.onnx')
save_model(model, input_image, 'Assets/Resources/model.onnx')

# # Test the model
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for i in range(training_image_count, image_count):
#         (input_image, _) = input_loader[i]
#         (train_image, _) = train_loader[i]

#         output_image = model(input_image)
#         _, predicted = torch.max(output_image.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')