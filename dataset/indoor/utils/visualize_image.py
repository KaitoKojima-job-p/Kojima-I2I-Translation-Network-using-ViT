
import numpy as np
import os
import h5py
from PIL import Image

def convert_hdf5_to_png(input_file, output_png_path):
    with h5py.File(input_file, 'r') as f:
        # 法線画像を平坦化して取得
        dataset_values = f['normals'][:]
        dataset_values = np.array(dataset_values)

        # float32->uint8に変換
        normalized_values = (dataset_values - np.min(dataset_values)) / (np.max(dataset_values) - np.min(dataset_values)) * 255
        normalized_values = normalized_values.astype(np.uint8)
        # np配列で元の次元に再構成
        rgb_data = np.stack([normalized_values[:,:,0], normalized_values[:,:,1], normalized_values[:,:,2]], axis=-1)
        # 法線カラー画像を保存する
        image = Image.fromarray(rgb_data)   
        output_normals_png_path = os.path.join(output_png_path, 'normals.png')
        image.save(output_normals_png_path)
        print(f"PNG file '{output_normals_png_path}' created successfully.")

        # カラー画像を取得し、保存する
        dataset_values = np.array(f['colors'])
        image = Image.fromarray(dataset_values)
        output_colors_png_path = os.path.join(output_png_path, 'colors.png')
        image.save(output_colors_png_path)
        print(f"PNG file '{output_colors_png_path}' created successfully.")

convert_hdf5_to_png(
                    input_file=r"C:\Users\kojik\program\source_test10\blender_code\output_dir\hdf5s\1.hdf5",
                    output_png_path=r"C:\Users\kojik\program\source_test10\blender_code"
                    )