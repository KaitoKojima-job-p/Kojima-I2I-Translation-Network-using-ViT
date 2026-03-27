import argparse
import os
import numpy as np
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

from models_vgg import *
from datasets import *
#from model_trans import *
# testの方のgeneratorを設定
from z_torch_test_ver4 import *

from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models 

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips as lpips_model

# 引数を設定する関数
# データセットや保存のパスを設定
# Adamのパラメータを設定
# cpuや画像サイズの指定

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="dataset\\original", help="name of the dataset to use for ML")
    parser.add_argument("--save_file_name", type=str, default="7", help="save file name")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    return parser.parse_args()

def main(opt):
    
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        print("CUDA is gpu")
    else:
        print("CUDA is not gpu")
    
    # Initialize generator and discriminator
    generator = Generator()

    if cuda:
        generator = generator.cuda()
    
    # Configure dataloaders
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    sample_dataloader = DataLoader(
        ImageDataset(opt.dataset_name, transforms_=transforms_, mode="sample"),
        batch_size=49,
        shuffle=None,
        num_workers=1,
        drop_last = True
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  

    for i, batch in enumerate(sample_dataloader):
        # Inputs
        real_A = Variable(batch["B"].type(Tensor)) 
        fake_B = generator(real_A)


if __name__ == '__main__':
    opt = get_args()
    main(opt)
    
