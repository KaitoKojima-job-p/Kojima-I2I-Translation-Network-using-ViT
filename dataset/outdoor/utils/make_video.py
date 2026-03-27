import cv2
import os
from glob import glob

def make_video_from_images(image_dir, output_path, fps=30, num_frames=100):
    # 画像ファイルを昇順で取得（jpg, png対応）
    image_paths = sorted(glob(os.path.join(image_dir, '*')))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_paths) < num_frames:
        raise ValueError(f"画像が {num_frames} 枚未満です。現在: {len(image_paths)}")

    # 最初の画像でサイズ取得
    sample_img = cv2.imread(image_paths[0])
    height, width, _ = sample_img.shape

    # 動画ライターを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'でmp4形式
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        img = cv2.imread(image_paths[i])
        if img is None:
            print(f"読み込めない画像: {image_paths[i]}")
            continue
        video_writer.write(img)

    video_writer.release()
    print(f"動画を保存しました: {output_path}")

# 使用例
make_video_from_images(
    image_dir= 'C:\\Users\kojik\\code\\program\\source_test22\\make_dateset\\output_image\\test_iriyama__',     # ← 画像の入ったフォルダに変更
    output_path= 'output_video2.mp4',
    fps=30,
    num_frames=300
)
