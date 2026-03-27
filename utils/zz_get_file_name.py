import os
import re

# 指定ディレクトリ
directory = r"C:\Users\kojik\program\source_test25\dataset\original\test_final_ver2"  # ここを適切なパスに変更
output_txt = "zz_output_test_final_ver2.txt"  # 出力するテキストファイル名

# ファイル名から name1_name2_ の部分を抽出する正規表現
pattern = re.compile(r"^(.*?_.*?)_\d{4}\.png$")

# 一意なプレフィックスを保存するためのセット
prefixes = set()

# ディレクトリ内のファイルを走査
for filename in os.listdir(directory):
    if filename.endswith(".png"):  # PNGファイルのみ処理
        parts = filename.rsplit("_", 1)  # 最後のアンダーバーで分割
        if len(parts) == 2 and parts[1][:-4].isdigit():  # `_xxxx.png` の形式を確認
            prefixes.add(parts[0])  # `_xxxx` の前の部分を保存

# 結果をテキストファイルに出力
with open(output_txt, "w") as f:
    for prefix in sorted(prefixes):  # ソートして出力（任意）
        f.write(prefix + "\n")

print(f"プレフィックスを {output_txt} に保存しました。")
