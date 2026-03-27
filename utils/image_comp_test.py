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

import re


def create_combined_image(original_image_paths, output_path="image_eval_test/images/combined_output.png"):
    combined_width = 256 * 10
    combined_height = 256 * 3
    combined_image = Image.new("RGB", (combined_width, combined_height))

    for i in range(3):
        img_path = original_image_paths[i % 3]
        if os.path.isfile(img_path):  # ファイルが存在するか確認
            img = Image.open(img_path).convert('RGB')

            for j in range(10):
                cropped_image = img.crop((256 * (j % 5), 256 * (j // 5), 256 * (j % 5 + 1), 256 * (j // 5 + 1))) 
                combined_image.paste(cropped_image, (j * 256, i * 256))

    # 画像を保存
    combined_image.save(output_path)

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

    transforms_ = [
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    val_dataloader = DataLoader(
        ImageDataset("dataset/original", transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
        drop_last = True
    )

    """Saves a generated sample from the validation set"""

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

    os.makedirs("image_eval_test/images/sample", exist_ok=True)

    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B, _ = generator_Trans(real_A)
    fake_B = torch.clip(fake_B,0,1)
    paths = imgs["path"]

    save_image(real_B.data, "image_eval_test/images/sample/Real_B_CNN.png", nrow=5, padding=0, normalize=True)
    save_image(fake_B.data, "image_eval_test/images/sample/fake_B_Trans.png", nrow=5, padding=0, normalize=True)

    imgs_B = (imgs["B"] - 0.5) / 0.5
    real_A = Variable(imgs_B.type(Tensor))
    fake_B = generator_CNN(real_A)

    save_image(fake_B.data, "image_eval_test/images/sample/fake_B_CNN.png", nrow=5, padding=0, normalize=True)

    original_image_paths = sorted(
        [os.path.join("image_eval_test/images/sample", filename)
         for filename in os.listdir("image_eval_test/images/sample")
         if filename.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(re.search(r'(\d+)', x).group()) if re.search(r'(\d+)', x) else x
    )[:3]
    print("使用する画像の順番:", original_image_paths)
    create_combined_image(original_image_paths=original_image_paths)

if __name__ == '__main__':
    main()