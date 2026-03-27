import os
import shutil

parent_dir = r"C:\Users\kojik\code\program\source_test29\3D-front\output_image"
output_dir = os.path.join(os.path.dirname(parent_dir), "all_output_image")

# output_dir作成（存在すれば中身クリア）
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

test_dir = os.path.join(output_dir, "test")
train_dir = os.path.join(output_dir, "train")
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# 昇順ソートでサブディレクトリ取得
subdirs = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])
test_subdirs = subdirs[:2]
train_subdirs = subdirs[2:]

print("TEST:", test_subdirs)
for subdir in test_subdirs:
    src_dir = os.path.join(parent_dir, subdir)
    
    # pngファイルをtest直下に平坦移動
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        if item.lower().endswith('.png'):
            dst_item = os.path.join(test_dir, item)
            shutil.move(src_item, dst_item)
    
    # 空ディレクトリ削除
    os.rmdir(src_dir)

print("TRAIN:", [d for d in train_subdirs])
for subdir in train_subdirs:
    src_dir = os.path.join(parent_dir, subdir)
    
    # pngファイルをtrain直下に平坦移動
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        if item.lower().endswith('.png'):
            dst_item = os.path.join(train_dir, item)
            shutil.move(src_item, dst_item)
    
    # 空ディレクトリ削除
    os.rmdir(src_dir)

print("完了")
