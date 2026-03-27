import os
from PIL import Image

def crop_grid_images(image_path, sub_dir_name):
    """
    7x7のグリッド画像から特定の座標を切り出し、
    関数名ディレクトリの下にサブディレクトリを作成して保存する。
    """
    # 1. 関数名のディレクトリ（ベースディレクトリ）を作成
    base_dir = r"C:\Users\kojik\code\program\source_test27\zz_crop_grid_images"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 2. 保存先のサブディレクトリを作成
    target_path = os.path.join(base_dir, sub_dir_name)
    os.makedirs(target_path, exist_ok=True)

    # 3. 画像の読み込み
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"エラー: 画像を開けませんでした。{e}")
        return

    # 1枚のサイズ
    size = 256

    # 4. 特定位置の切り出し
    # PILのcrop引数は (left, upper, right, lower)
    
    # 7行2列目 (インデックス: row=6, col=1)
    # y: 6*256=1536, x: 1*256=256
    box1 = (256, 1536, 512, 1792)
    img1 = img.crop(box1)
    
    # 6行5列目 (インデックス: row=5, col=4)
    # y: 5*256=1280, x: 4*256=1024
    box2 = (1024, 1280, 1280, 1536)
    img2 = img.crop(box2)

    # 5. 画像の保存
    img1.save(os.path.join(target_path, "1.png"))
    img2.save(os.path.join(target_path, "2.png"))
    
    print(f"保存完了: {target_path}")

def main():
    # 入力画像と保存先ディレクトリ名を指定
    input_file = r"C:\Users\kojik\code\program\source_test27\result\indoor_256_2_vgg025\samples\100.png"  # 実際のファイル名に変更してください
    save_subdir = "indoor_256_2_vgg025"
    
    crop_grid_images(input_file, save_subdir)

if __name__ == "__main__":
    main()