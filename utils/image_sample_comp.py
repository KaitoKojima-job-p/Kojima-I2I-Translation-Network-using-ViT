import argparse
import os
import numpy as np
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

from models_vgg import *
from datasets import *
#from model_trans import *
# testの方のgeneratorを設定
from z_torch_test import *

from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch


def main():
    os.makedirs("image_eval_test/images", exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        print("CUDA is gpu")
    else:
        print("CUDA is not gpu")

    generator_CNN = GeneratorUNet()
    generator_Trans = Generator()


    if cuda:
        generator_CNN = generator_CNN.cuda()
        generator_Trans = generator_Trans.cuda()

    generator_CNN.load_state_dict(torch.load("image_eval_test/model/CNN_generator_100.pth"))
    generator_Trans.load_state_dict(torch.load("image_eval_test/model/Trans_generator_100.pth"))

    generator_CNN.eval()
    generator_Trans.eval()

    transforms_CNN = [
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transforms_Trans = [
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    val_dataloader_CNN = DataLoader(
        ImageDataset("image_eval_test/dataset/original", transforms_=transforms_CNN, mode="val"),
        batch_size=4,
        shuffle=False,
        num_workers=1,
        drop_last = True
    )

    """Saves a generated sample from the validation set"""

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

    try:
        imgs = next(iter(val_dataloader_CNN))
    except StopIteration:
        print("CNN DataLoader has reached the end of the dataset.")
        return
    except Exception as e:
        print(f"CNN An error occurred: {e}")
        return

    real_A_CNN = Variable(imgs["B"].type(Tensor))
    real_B_CNN = Variable(imgs["A"].type(Tensor))

    imgs_B_normalized = (imgs["B"] + 1) / 2
    real_A_Trans =Variable(imgs_B_normalized.type(Tensor))

    fake_B_CNN = generator_CNN(real_A_CNN)
    fake_B_Trans, _ = generator_Trans(real_A_Trans)
    fake_B_Trans_max = fake_B_Trans.max().item()
    fake_B_Trans_min = fake_B_Trans.min().item()
    fake_B_Trans = 2 * (fake_B_Trans - fake_B_Trans_min) / (fake_B_Trans_max - fake_B_Trans_min) - 1

    img_sample = torch.cat((fake_B_CNN.data, fake_B_Trans.data, real_B_CNN.data), -2)
    save_image(img_sample, "image_eval_test/images/1.png", nrow=4, padding=0, normalize=True)


if __name__ == '__main__':
    main()