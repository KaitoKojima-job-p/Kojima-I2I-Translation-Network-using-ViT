import os
import shutil
import random

def copy_random_json_files(src_dir, dst_dir, num_files):
    # src_dirから.jsonファイルをすべて取得
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]

    if len(json_files) == 0:
        print("指定されたディレクトリにjsonファイルがありません")
        return

    # dst_dirにすでにあるjsonファイルを取得
    existing_files = set()
    if os.path.exists(dst_dir):
        existing_files = set(f for f in os.listdir(dst_dir) if f.endswith('.json'))

    # コピー対象の候補ファイル（まだコピーされていないもの）
    available_files = [f for f in json_files if f not in existing_files]

    if len(available_files) == 0:
        print("コピーできる新しいjsonファイルがありません")
        return

    # コピー個数は候補より多ければ調整
    num_files = min(num_files, len(available_files))

    selected_files = []
    tries = 0
    max_tries = 10  # 無限ループを防ぐリトライ回数制限

    while len(selected_files) < num_files and tries < max_tries:
        needed = num_files - len(selected_files)
        # 候補からランダム選択（重複なし）
        pick = random.sample(available_files, needed)
        
        # 選んだものが既に選択済みでなければ追加
        for f in pick:
            if f not in selected_files:
                selected_files.append(f)
        
        # 再検討用にavailable_filesから選択済みを除外
        available_files = [f for f in available_files if f not in selected_files]
        tries += 1

    if len(selected_files) < num_files:
        print(f"注意: {num_files}個中{len(selected_files)}個しか新しいファイルがありませんでした。")

    # コピー先ディレクトリがなければ作成
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 選択ファイルをコピー
    for file_name in selected_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {file_name} to {dst_dir}")


# 使い方例
src_directory = r"D:\programD\datasets\3dfront\3D-FRONT"
dst_directory = r"C:\Users\kojik\code\program\source_test29\3D-front\jsons\test_zz"

os.makedirs(dst_directory, exist_ok=True)

num_files_to_copy = 1
copy_random_json_files(src_directory, dst_directory, num_files_to_copy)
