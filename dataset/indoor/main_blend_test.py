import blenderproc as bproc
import argparse
import os
import math
import numpy as np
import h5py
from PIL import Image
import sys

# コマンドライン引数の強制指定用のコード（今回は直接値を渡すため一部変更）
class Args:
    front = "D:/programD/datasets/3dfront/3D-FRONT/e1c6b002-7d08-437d-8981-9642be46f5c0.json"
    future_folder = "D:/programD/datasets/3dfront/3D-FUTURE-model"  # ※適宜修正してください
    front_3D_texture_path = "D:/programD/datasets/3dfront/3D-FRONT-texture"  # ※適宜修正してください
    output_dir = "C:/Users/kojik/code/program/source_test29/3D-front/output_hdf5"  # hdf5出力先（適宜修正可）
    blend_save_dir = "C:/Users/kojik/code/program/source_test29/3D-front/blends/test_zz"  # blend保存先


args = Args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                 transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0

def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False

# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]

location = np.array([1.4491, 5.7116, 1.6339])
rotation = np.array([
    math.radians(72.433),   # X
    math.radians(0.0),      # Y
    math.radians(96.575)    # Z
])

cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
bproc.camera.add_camera_pose(cam2world_matrix)

# Also render normals
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id"])

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)


def convert_hdf5_to_png(input_file, output_png_path):
    os.makedirs(output_png_path, exist_ok=True)
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

input = os.path.join(args.output_dir, "0.hdf5")
convert_hdf5_to_png(
                    input_file=input,
                    output_png_path= "C:/Users/kojik/code/program/source_test29/3D-front/output_image"
                    )

# -------------------------------------------
# **ここからblendファイル保存の追加コード**

# blend保存先ディレクトリがなければ作成
if not os.path.exists(args.blend_save_dir):
    os.makedirs(args.blend_save_dir)

# jsonファイル名拡張子なしを使いblendファイル名生成
json_base_name = os.path.splitext(os.path.basename(args.front))[0]
blend_save_path = os.path.join(args.blend_save_dir, f"{json_base_name}.blend")

# 現在のBlenderシーンをblendファイルとして保存
import bpy
bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)
print(f"Blendファイルを保存しました: {blend_save_path}")
