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
from z_torch_test_ver2 import *

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
    parser.add_argument("--dataset_name", type=str, default="C:\\Users\\kojik\\code\\program\\source_test29\\3D-front\\all_output_image", help="name of the dataset to use for ML")
    parser.add_argument("--save_file_name", type=str, default="indoor_256_1", help="save file name")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate of generator")
    parser.add_argument("--lr_d", type=float, default=0.0001, help="adam: learning rate of discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=2, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    return parser.parse_args()


# 評価関数の定義
def evaluate_metrics(generator, val_dataloader, Tensor, fid, lpips):
    def normalize_tensor(tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)  # 小さい値を足して安定化
        return normalized_tensor

    generator.eval()  # 評価モードに設定

    num_samples = 0
    psnr_total = 0
    ssim_total = 0
    lpips_total = 0

    with torch.no_grad():
        for imgs in val_dataloader:
            real_A = Variable(imgs["B"].type(Tensor))
            real_B = Variable(imgs["A"].type(Tensor))
            fake_B = generator(real_A)

            real_B = normalize_tensor(real_B)
            fake_B = normalize_tensor(fake_B)

            data_batch = real_A.size(0)
            num_samples += 1

            ### caluculate PSNR ###
            psnr_batch = 0
            for i in range(data_batch):
                psnr_batch += psnr(real_B[i].cpu().numpy().transpose(1, 2, 0), 
                                   fake_B[i].cpu().numpy().transpose(1, 2, 0))
            psnr_total += psnr_batch / data_batch
            
            ### caluculate SSIM ###
            ssim_batch = 0
            for i in range(data_batch):
                ssim_batch += ssim(real_B[i].cpu().numpy().transpose(1, 2, 0), 
                                   fake_B[i].cpu().numpy().transpose(1, 2, 0), 
                                   data_range=1.0, multichannel=True, channel_axis=2)
            ssim_total += ssim_batch / data_batch

            ### caluculate LPIPS ###
            lpips_batch = 0.0
            for i in range(data_batch):
                # LPIPSスコアの計算
                lpips_score = lpips(real_B[i].unsqueeze(0), fake_B[i].unsqueeze(0))
                lpips_batch += lpips_score.item()
            lpips_total += lpips_batch / data_batch

            ## caluculate FID ###
            fid_score = 0
    
    psnr_average = psnr_total / num_samples
    ssim_average = ssim_total / num_samples
    lpips_average = lpips_total / num_samples 

    generator.train()

    return psnr_average, ssim_average, lpips_average, fid_score


def evaluate_and_save_metrics(epoch, 
                              generator, 
                              val_dataloader,
                              Tensor, fid, lpips):
    psnr, ssim, lpips, fid = evaluate_metrics(generator, 
                                              val_dataloader, 
                                              Tensor, fid, lpips)

    with open(f"result/{opt.save_file_name}/evaluation/evaluation.csv", 'a') as f:
        f.write("Epoch, %d,psnr, %f,ssim, %f,lpips, %f,fid, %f\r" 
                % (
                    epoch, 
                    psnr, 
                    ssim,
                    lpips,
                    fid
                  )
                )


def main(opt):
    
    os.makedirs("result/%s/images" % opt.save_file_name, exist_ok=True)
    os.makedirs("result/%s/samples" % opt.save_file_name, exist_ok=True)
    os.makedirs("result/%s/saved_models" % opt.save_file_name, exist_ok=True)
    os.makedirs("result/%s/loss" % opt.save_file_name, exist_ok=True)
    os.makedirs("result/%s/evaluation" % opt.save_file_name, exist_ok=True)
    
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        print("CUDA is gpu")
    else:
        print("CUDA is not gpu")
    
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    
    # setttings of lambda
    ratio = 5.0
    lambda_vgg = 0.25
    lambda_pixel = 0
    lamda_PV = 1.0
    discriminator_ratio = 1
    
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
    
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator() 
    vgg19 = Vgg19_12345()
    lpips = lpips_model.LPIPS(net='alex')
    fid = FrechetInceptionDistance(feature=64, normalize=True)  # 64次元のInception特徴


    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        vgg19 = vgg19.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        lpips.cuda()
        fid.cuda()
    
    if opt.epoch > opt.checkpoint_interval:
        # Load pretrained models
        load_number = opt.epoch - (opt.epoch % opt.checkpoint_interval)
        if opt.epoch % opt.checkpoint_interval == 0:
            load_number = load_number - opt.checkpoint_interval
        print("load gen and dis No." + str(load_number)) 
        generator.load_state_dict(torch.load("result/%s/saved_models/generator_%d.pth" % (opt.save_file_name, load_number)))
        discriminator.load_state_dict(torch.load("result/%s/saved_models/discriminator_%d.pth" % (opt.save_file_name, load_number)))
    else:
        # Initialize weights
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
    
    # Configure dataloaders
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    dataloader = DataLoader(
        ImageDataset(opt.dataset_name, transforms_=transforms_, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        drop_last = True
    )
    
    val_dataloader = DataLoader(
        ImageDataset(opt.dataset_name, transforms_=transforms_, mode="test"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
        drop_last = True
    )

    sample_dataloader = DataLoader(
        ImageDataset(opt.dataset_name, transforms_=transforms_, mode="sample"),
        batch_size=49,
        shuffle=None,
        num_workers=1,
        drop_last = True
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  

    start_time = time.time()

    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs+1):

        total_adv_loss = 0
        total_pixel_loss = 0
        total_vgg_loss = 0
        total_PV_loss = 0
        total_gen_loss = 0
        total_dis_loss = 0
        
        for i, batch in enumerate(dataloader):

            # Inputs
            real_A = Variable(batch["B"].type(Tensor)) #input nomal image
            real_B = Variable(batch["A"].type(Tensor)) #input color image

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = 0

            # VGG loss
            loss_vgg = 0.0
            # get vgg feature
            fake_B_features = vgg19(fake_B)
            real_A_features = vgg19(real_B)
            # calculation vgg loss
            for j in range(5):
                loss_vgg += criterion_pixelwise(fake_B_features[j], real_A_features[j])

            # Pixel-wise and VGG composite loss
            loss_PV = (lambda_pixel * loss_pixel + lambda_vgg * loss_vgg) * lamda_PV
            
            # Total loss
            loss_G = loss_GAN + loss_PV

            loss_G.backward()

            optimizer_G.step()
 
            # ---------------------
            #  Train Discriminator
            # ---------------------

            if (epoch-1) % discriminator_ratio == 0 or epoch == opt.n_epochs:

                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(real_B, real_A)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(fake_B.detach(), real_A)
                loss_fake = criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()

                optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------     
            
            #calucurate total loss
            total_adv_loss += loss_GAN
            total_pixel_loss += loss_pixel
            total_vgg_loss += loss_vgg
            total_PV_loss += loss_PV
            total_gen_loss += loss_G
            if (epoch-1) % discriminator_ratio == 0 or epoch == opt.n_epochs:
                total_dis_loss += loss_D
            
            # Determine approximate time left
            batches_done = (epoch-1) * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            time_left = format_time_left(time_left)
            prev_time = time.time()
            
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [adv: %f, PV: %f, G loss: %f, D loss: %f,] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_GAN.item(),
                    loss_PV.item(),
                    loss_G.item(),
                    loss_D.item() if (epoch-1) % discriminator_ratio == 0 or epoch == opt.n_epochs else 0,
                    time_left,
                )
            )
        
        ave_adv_loss = total_adv_loss / len(dataloader)
        ave_pixel_loss = total_pixel_loss / len(dataloader)
        ave_vgg_loss = total_vgg_loss / len(dataloader)
        ave_PV_loss = total_PV_loss / len(dataloader)
        ave_gen_loss = total_gen_loss / len(dataloader)
        if (epoch-1) % discriminator_ratio == 0 or epoch == opt.n_epochs:
            ave_dis_loss = total_dis_loss / len(dataloader)

        with open(f'result/{opt.save_file_name}/loss/loss.csv', 'a') as f:
            f.write("Epoch, %d/%d, adv , %f, pixel , %f, vgg , %f, PV, %f, G, %f, D, %f\r"
                    % (
                        epoch,
                        opt.n_epochs,
                        ave_adv_loss.item(),
                        #ave_pixel_loss.item(),
                        ave_pixel_loss,
                        ave_vgg_loss.item(),
                        ave_PV_loss.item(),
                        ave_gen_loss.item(),
                        ave_dis_loss.item() if (epoch-1) % discriminator_ratio == 0 or epoch == opt.n_epochs else 0,
                        )
                    )

        # save image of loss-plot    
        plot_losses_and_save(f"result/{opt.save_file_name}/loss/loss.csv")
            
        # save sample image
        if epoch % opt.sample_interval == 0:
            generator.eval()
            sample_images(generator, val_dataloader, epoch, opt.save_file_name, Tensor)
            generator.train()
        
        # Save model
        if epoch % opt.checkpoint_interval == 0:
            evaluate_and_save_metrics(epoch, 
                                      generator, 
                                      val_dataloader, 
                                      Tensor, fid, lpips)
            plot_metrics_and_save(f"result/{opt.save_file_name}/evaluation/evaluation.csv")
            generator.eval()
            sample_images_cp(generator, sample_dataloader, epoch, opt.save_file_name, Tensor)
            generator.train()

            torch.save(generator.state_dict(), "result/%s/saved_models/generator_%d.pth" % (opt.save_file_name, epoch))
            torch.save(discriminator.state_dict(), "result/%s/saved_models/discriminator_%d.pth" % (opt.save_file_name, epoch))

        # 最終ネットワークと、一日ごとにネットワークを保存
        """
        cur_time = time.time()
        if epoch == (opt.n_epochs+1-opt.epoch) or cur_time - start_time >= 86400:
            torch.save(generator.state_dict(), "result/%s/saved_models/generator_%d.pth" % (opt.save_file_name, epoch))
            torch.save(discriminator.state_dict(), "result/%s/saved_models/discriminator_%d.pth" % (opt.save_file_name, epoch))
            
            start_time = time.time()
        """


if __name__ == '__main__':
    opt = get_args()
    main(opt)
    
