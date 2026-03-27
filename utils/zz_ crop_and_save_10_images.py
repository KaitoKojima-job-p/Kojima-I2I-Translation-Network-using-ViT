import os
from PIL import Image

def crop_and_save_10_samples(input_path, output_base_dir):
    """
    画像の並び順:
    Row 0-2: Sample 1-5 (input, output, target)
    Row 3-5: Sample 6-10 (input, output, target)
    """
    
    # 親ディレクトリ作成
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 入力画像読み込み
    img = Image.open(input_path)
    width, height = img.size
    print(f"入力画像サイズ: {width} x {height}")
    
    # タイルサイズの計算
    tile_width = width // 5   # 横に5枚
    tile_height = height // 6  # 縦に6行 (1-5の3行 + 6-10の3行)
    print(f"1枚あたりのサイズ: {tile_width} x {tile_height}")
    
    # 1番から10番までのサンプルを順に処理
    for sample_idx in range(1, 11):
        # 保存先ディレクトリ作成 (subdir_01, subdir_02...)
        subdir_path = os.path.join(output_base_dir, f"sample_{sample_idx:02d}")
        os.makedirs(subdir_path, exist_ok=True)
        
        # ブロック（上下）の判定
        if sample_idx <= 5:
            block_row_offset = 0  # 0, 1, 2行目を使用
            col_idx = sample_idx - 1
        else:
            block_row_offset = 3  # 3, 4, 5行目を使用
            col_idx = sample_idx - 6
            
        # 各サンプルの3つのドメインを切り出し
        # 0: input (Real A), 1: output (Fake B), 2: target (Real B)
        domains = ["input", "output", "target"]
        
        for i, domain_name in enumerate(domains):
            current_row = block_row_offset + i
            
            left = col_idx * tile_width
            top = current_row * tile_height
            right = left + tile_width
            bottom = top + tile_height
            
            cropped_img = img.crop((left, top, right, bottom))
            
            # 保存
            output_path = os.path.join(subdir_path, f"{domain_name}.png")
            cropped_img.save(output_path)
            
        print(f"サンプル {sample_idx:02d} を保存しました。")

    print("\nすべての切り出しが完了しました！")

def main():
    # パスはご自身の環境に合わせて適宜変更してください
    input_path = r"C:\Users\kojik\code\program\source_test27\result\indoor_256_1\images\100.png"
    output_base_dir = r"C:\Users\kojik\code\program\source_test27\zz_crop_and_save_10_samples"

    crop_and_save_10_samples(input_path, output_base_dir)

if __name__ == "__main__":
    main()