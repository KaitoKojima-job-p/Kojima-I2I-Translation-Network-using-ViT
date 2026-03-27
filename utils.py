from torchvision.utils import save_image
from torch.autograd import Variable
import torch

import csv
import matplotlib.pyplot as plt
import os


# サンプル画像の保存する関数
def sample_images(generator, val_dataloader, epoch, save_file_name, Tensor):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "result/%s/images/%s.png" % (save_file_name, epoch), nrow=5, padding=0, normalize=True)


def sample_images_cp(generator, sample_dataloader, epoch, save_file_name, Tensor):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(sample_dataloader))
    real_A = Variable(imgs["B"].type(Tensor))
    fake_B = generator(real_A)
    save_image(fake_B.data, "result/%s/samples/%s.png" % (save_file_name, epoch), nrow=7, padding=0, normalize=True)

    if epoch == 10:
        real_B = Variable(imgs["A"].type(Tensor))
        save_image(real_B.data, "result/%s/samples/0_GT.png" % (save_file_name), nrow=7, padding=0, normalize=True)




# 残り時間のフォーマットを変更する関数
def format_time_left(time_left):
    """
    残り時間を '日, 時間:分' 形式の文字列にフォーマットする関数
    
    :param time_left: datetime.timedelta オブジェクト
    :return: フォーマットされた文字列
    """
    days = time_left.days
    hours, remainder = divmod(time_left.seconds, 3600)
    minutes = remainder // 60
    
    # フォーマットされた時間文字列を作成
    return f"{days}d:{hours:02}:{minutes:02}"


def plot_metrics_and_save(csv_file_path):
    epochs = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    fid_values = []

    # CSVファイルを読み込む
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) >= 10:  # 1, 3, 5 のインデックスを処理するために最低6列が必要
                epoch = int(line[1].strip())
                psnr = float(line[3].strip())
                ssim = float(line[5].strip())
                lpips = float(line[7].strip())
                fid = float(line[9].strip())
                epochs.append(epoch)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                lpips_values.append(lpips)
                fid_values.append(fid)

    # グラフをプロット
    plt.figure(figsize=(16, 12))

    # PSNRグラフ
    plt.subplot(2, 2, 1)
    plt.plot(epochs, psnr_values, marker='o', linestyle='-', color='r', label='PSNR')
    plt.title('PSNR Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()

    # SSIMグラフ
    plt.subplot(2, 2, 2)
    plt.plot(epochs, ssim_values, marker='o', linestyle='-', color='b', label='SSIM')
    plt.title('SSIM Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    # LPIPSグラフ
    plt.subplot(2, 2, 3)
    plt.plot(epochs, lpips_values, marker='o', linestyle='-', color='g', label='LPIPS')
    plt.title('LPIPS Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.legend()

    # FIDグラフ
    plt.subplot(2, 2, 4)
    plt.plot(epochs, fid_values, marker='o', linestyle='-', color='m', label='FID')
    plt.title('FID Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.legend()

    # グラフ画像の保存
    graph_image_path = os.path.join(os.path.dirname(csv_file_path), 'metrics_graph.png')
    plt.tight_layout()
    plt.savefig(graph_image_path)
    plt.close()


# 損失のcsvファイルのグラフを画像で保存する関数
def plot_losses_and_save(csv_file_path):

    def plot_loss_subplot(epochs, losses, title, ylabel, color, subplot_position):
        """損失データをサブプロットにプロットする"""
        plt.subplot(subplot_position)
        plt.plot(epochs, losses, marker='o', linestyle='-', color=color, label=title)
        plt.title(f'{title} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()

    epochs = []
    all_losses = [[] for _ in range(6)]  # 6つの損失を格納するリスト

    # CSVファイルを読み込む
    with open(csv_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 11:
                epoch = int(parts[1].split('/')[0].strip())
                losses = [float(parts[i].strip()) for i in [3, 5, 7, 9, 11, 13]]
                # D Loss（discriminatorの損失）が0の場合、その行のD Lossを無視
                if losses[5] == 0.0:
                    losses[5] = None  # 0を無視するためNoneを設定

                epochs.append(epoch)
                for i in range(6):
                    all_losses[i].append(losses[i])

    # 各損失をプロット
    plt.figure(figsize=(12, 12))

    titles = ['Adversarial Loss', 'Pixel Loss', 'VGG Loss', 'PV Loss', 'G Loss', 'D Loss']
    colors = ['g', 'r', 'b', 'm', 'c', 'y']
    for i, (losses, title, color) in enumerate(zip(all_losses, titles, colors)):
        # D LossがNoneの箇所をフィルタリング
        filtered_epochs = [e for e, l in zip(epochs, losses) if l is not None]
        filtered_losses = [l for l in losses if l is not None]

        if filtered_losses:  # フィルタリング後にデータが残っている場合のみプロット
            plot_loss_subplot(filtered_epochs, filtered_losses, title, title, color, 321 + i)

    plt.tight_layout()

    # グラフ画像の保存
    graph_image_path = os.path.join(os.path.dirname(csv_file_path), 'losses_graph.png')
    plt.savefig(graph_image_path)
    plt.close()
