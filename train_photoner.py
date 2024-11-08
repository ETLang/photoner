import os
import sys
from PIL import Image
import re

import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision as tv
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from vgg import Vgg16

from photoner_net import PhotonerNet

# Training Procedure!
# Load Dataset
# Cull Crappy Samples
# Treat each color channel as its own dataset (algorithm is the same for all colors)
# Set up model for training
# Set up loss algorithm
# Train!
# Test!
# Once it works well, optimize for real time use.
# Save ONNX
# Set up unity project to utilize ONNX model
# Get UE5 source code and dig in!

def do_photon_estimate(model, image, kernel_size):
    (_,_,h,w) = image.size()

    padder = transforms.Pad(kernel_size // 2, padding_mode="reflect")

    image = padder(image)
    output = torch.ones(1, 3, h, w).cpu()

    output = torch.Tensor()

    wTrimmed = w - prescale_chunk_size*2
    hTrimmed = h - prescale_chunk_size*2
    for x in range(0, wTrimmed, prescale_chunk_size):
        col = torch.Tensor()
        for y in range(0, hTrimmed, prescale_chunk_size):
            sub = image[:, :, (y):(y+prescale_chunk_size*2), (x):(x+prescale_chunk_size*2)]
            return model.sub()
            #output[:,:,scale*y:(scale*y+chunk_size), scale*x:(scale*x+chunk_size)] = model(sub).cpu()
            col = torch.cat(col, model(sub), 2)

        if h % prescale_chunk_size != 0:
            y = (h - prescale_chunk_size*2)
            sub = image[:, :, y:h, (x):(x+prescale_chunk_size*2)]
            #output[:,:,scale*y:(scale*y+chunk_size), scale*x:(scale*x+chunk_size)] = model(sub).cpu()
            col = torch.cat(col, model(sub), 2)

        output = torch.cat(output, col, 3)

    if w % prescale_chunk_size != 0:
        col = torch.Tensor()
        x = (w - prescale_chunk_size*2)

        for y in range(0, h, prescale_chunk_size):
            sub = image[:, :, (y):(y+prescale_chunk_size*2), x:w]
            #output[:,:,scale*y:(scale*y+chunk_size), scale*x:(scale*x+chunk_size)] = model(sub).cpu()
            col = torch.cat(col, model(sub), 2)

        if h % prescale_chunk_size != 0:
            y = (h - prescale_chunk_size*2)
            sub = image[:, :, y:h, x:w]
            #output[:,:,scale*y:(scale*y+chunk_size), scale*x:(scale*x+chunk_size)] = model(sub).cpu()
            col = torch.cat(col, model(sub), 2)

        output = torch.cat(output, col, 3)


    return output

def train(args):
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        #transforms.RandomCrop(args.chunk_size * 2, padding=args.chunk_size // 2, pad_if_needed=True, padding_mode="reflect"),
        transforms.RandomCrop(args.chunk_size * args.super_scale, padding=args.chunk_size * args.super_scale // 4, pad_if_needed=True, padding_mode="reflect"),
        transforms.ToTensor()
        # transforms.Lambda(lambda x: x.mul(255))
    ])

    #resize_transform = transforms.Resize(args.chunk_size * 2 // args.super_scale, transforms.InterpolationMode.BICUBIC)
    resize_transform = transforms.Resize(args.chunk_size * 2, transforms.InterpolationMode.BICUBIC)
    select_center_transform = transforms.CenterCrop(args.chunk_size * args.super_scale)

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=1)

    transformer = MySupersamplingNet(args.chunk_size, args.super_scale).to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)

    sample_image = None
    if(args.checkpoint_sample is not None):
        sample_image = Image.open(args.checkpoint_sample).convert('RGB')
        sample_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Lambda(lambda x: x.mul(255))
        ])
        sample_image = sample_transform(sample_image)
        sample_image = sample_image.unsqueeze(0).to(device)

    for epoch in range(args.epochs):
        transformer.train()
        count = 0
        total_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            count += 1
            x = x.to(device)
            optimizer.zero_grad()

            # Downsize the image 4x with bicubic resampling
            x_small = resize_transform(x)

            y = transformer(x_small)

            x = normalize_batch(x)
            y = normalize_batch(y)

            expect = select_center_transform(x)
            actual = y

            features_expect = vgg(expect)
            features_actual = vgg(actual)

            loss = mse_loss(features_expect.relu2_2, features_actual.relu2_2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if count % args.checkpoint_interval == 0:
                msg = "[{}:{}] - Loss: {}".format(epoch, count, total_loss / count)
                print(msg)
                if args.checkpoint_model_dir is not None:
                    transformer.eval()

                    if(sample_image is not None):
                        sample_filename = "cp_" + str(epoch) + "_" + str(idx + 1) + "_sample.png"
                        sample_path = os.path.join(args.checkpoint_model_dir, sample_filename)
                        sample_output = do_supersample(transformer, sample_image, args.chunk_size, args.super_scale);
                        save_image(sample_path, sample_output[0])
                        # show_image(sample_image)
                        # show_image(sample_output)
                        

                    transformer.cpu()
                    cp_model_filename = "cp_" + str(epoch) + "_" + str(idx + 1) + "_model.pth"
                    cp_model_path = os.path.join(args.checkpoint_model_dir, cp_model_filename)
                    torch.save(transformer.state_dict(), cp_model_path)
                    transformer.to(device).train()

def photon_estimate(args):
    print("estimating photon field")
    device = torch.device("cuda" if args.cuda else "cpu")
    input_image = Image.open(args.image).convert('RGB')

    input_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    input_image = input_transform(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    with torch.no_grad():
        model = PhotonerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        output = do_photon_estimate(model, input_image, args.kernel_size)

        # if args.export_onnx:
        #     assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
        #     output = torch.onnx._export(
        #         style_model, content_image, args.export_onnx, opset_version=11,
        #     ).cpu()            
        # else:
        #     output = style_model(content_image).cpu()

    save_image(args.output, output[0])
    # Slice the input image for processing

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def save_image(filename, data):
    img = transforms.Lambda(lambda x: x.mul(255))(data).cpu().detach().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def show_image(tensor_image):
    plt.imshow(tensor_image[0].permute(1, 2, 0).cpu().detach())
    plt.waitforbuttonpress() 

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for my-ss")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="Interval between save checkpoints, default is 500")
    train_arg_parser.add_argument("--checkpoint-sample", type=str, default=None,
                                  help="path to image to evaluate at checkpoints")
    train_arg_parser.add_argument("--kernel-size", type=int, default=64,
                                  help="size of network kernel, default is 64")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/scaling arguments")
    eval_arg_parser.add_argument("--image", type=str, required=True,
                                 help="path to photon to smooth")
    eval_arg_parser.add_argument("--output", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for supersampling the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        photon_estimate(args)

if __name__ == "__main__":
    main()
