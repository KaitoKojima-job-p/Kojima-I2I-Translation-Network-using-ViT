import blenderproc as bproc
import bpy
import numpy as np
import os
import sys
import h5py
from PIL import Image
import math

from mathutils import Vector, Matrix, Euler, Quaternion


def set_camera():
    r = np.random.uniform(0.25, 0.5)
    theta = np.random.uniform(0, 3/4*np.pi)
    phi = np.random.uniform(0, 2*np.pi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    rx = theta
    ry = 0
    rz = phi + (np.pi / 2)

    return (x, y, z), (rx, ry, rz), r
 

def set_scene(input_hdri_path, output_dir, num_shots):
    c = bpy.context
    scn = c.scene

    c.scene.world.cycles_visibility.camera = True
    c.scene.world.cycles_visibility.Diffuse = True
    c.scene.world.cycles_visibility.Glossy = True
    c.scene.world.cycles_visibility.Transmission = True
    c.scene.world.cycles_visibility.Volume_Scatter = True

    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    hdri_directory_path = input_hdri_path
    hdri_files = [f for f in os.listdir(hdri_directory_path) if f.endswith('.hdr')]


    # load the objects into the scene
    obj_path =r"C:\Users\kojik\program\source_test22\make_dateset\object_data\stanford-bunny.obj"
    objs = bproc.loader.load_obj(obj_path)
    bpy.ops.object.origin_set()

    for obj in bpy.data.objects:
        if (obj.name == 'stanford-bunny'):
            bunny = obj
            e = obj.rotation_euler.copy()

    metallic = 1.0
    specular = 1.0
    roughness = 0.2

    material = bpy.data.materials.new(c.object.name)
    material.use_nodes = True
    m1_bsdf = material.node_tree.nodes['Principled BSDF']
    m1_bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
    m1_bsdf.inputs["Metallic"].default_value = metallic
    m1_bsdf.inputs["Specular"].default_value = specular
    m1_bsdf.inputs["Roughness"].default_value = roughness
    c.object.data.materials.append(material)
    c.object.data.materials[0] = material 


    tree_nodes.clear()
    node_background = tree_nodes.new(type='ShaderNodeBackground')
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    node_environment.location = -300,0
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
    node_output.location = 200,0
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        
    bproc.camera.set_resolution(256, 256)

    for hdri_file in sorted(hdri_files):
        hdr_path = os.path.join(hdri_directory_path, hdri_file)
        node_environment.image = bpy.data.images.load(hdr_path)

        # HDRIファイル名を取得しディレクトリ名を作成
        hdr_name = os.path.splitext(os.path.basename(hdri_file))[0]
        hdr_name_sanitized = hdr_name.replace(" ", "_")  # 必要に応じて加工
        save_dir = os.path.join(output_dir, hdr_name_sanitized)
        os.makedirs(save_dir, exist_ok=True)

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = 0

        for k in range(num_shots):
            z_rotation = np.random.uniform(0, np.pi/2)
            bunny.rotation_euler = (e[0], e[1], e[2]+z_rotation)
            bunny.keyframe_insert(data_path="rotation_euler", frame=k)
            #print(bunny.rotation_euler)

            bunny.location = (0, 0, 0)
            bunny.keyframe_insert(data_path="location", frame=k)

            camera_location, camera_rotation, r = set_camera()

            camera_obj = bpy.context.scene.camera
            cam_pose = bproc.math.build_transformation_mat(camera_location, camera_rotation)
            camera_obj.matrix_world = Matrix(cam_pose)

            camera_obj.rotation_mode = 'QUATERNION'

            # Persist camera pose
            camera_obj.keyframe_insert(data_path='location', frame=k)
            camera_obj.keyframe_insert(data_path='rotation_euler', frame=k)

            # カメラをさかさまに
            forward_direction = camera_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
            forward_direction.normalize()  # 正規化

            # ローカルZ軸を中心に指定した角度で回転するクォータニオンを生成
            random_rotation_rad = np.random.uniform(-np.pi/6, np.pi/6)
            rotation_quat = Quaternion(forward_direction,  random_rotation_rad)

            # 現在のカメラの回転を取得
            current_rotation = camera_obj.rotation_quaternion

            # 新しい回転を適用（現在の回転に対して回転を合成）
            new_rotation = rotation_quat @ current_rotation
            camera_obj.rotation_quaternion = new_rotation

            camera_obj.keyframe_insert(data_path='rotation_quaternion', frame=k)

            bpy.context.scene.frame_end = bpy.context.scene.frame_end + 1


        # 法線出力を有効化してレンダリング
        bproc.renderer.enable_normals_output()
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(save_dir, data)
        bpy.context.scene.frame_start = bpy.context.scene.frame_end

    bproc.clean_up


def convert_hdf5_to_png(input_dir, output_image_dir, num_shots):
    # 入力ディレクトリ内のサブディレクトリを処理
    for i, hdri_dir in enumerate(os.listdir(input_dir)):  # enumerate でインデックスを取得
        hdri_path = os.path.join(input_dir, hdri_dir)
        if os.path.isdir(hdri_path):  # ディレクトリかどうかを確認
            for j in range(num_shots):
                input_file = os.path.join(hdri_path, f"{j}.hdf5")
                if os.path.isfile(input_file):
                    with h5py.File(input_file, 'r') as f:
                        # カラー画像の読み込み
                        dataset_values_colors = np.array(f['colors'])
                        image_colors = Image.fromarray(dataset_values_colors)

                        # 法線画像の読み込みと処理
                        dataset_values_normals = np.array(f['normals'])
                        threshold = 0.20
                        zero_mask = (dataset_values_normals == 0.5)
                        zero_pixels_indices = np.where(zero_mask.all(axis=2))
                        near_zero_mask = (dataset_values_normals >= 0.5 - threshold) & (dataset_values_normals <= 0.5 + threshold) & (dataset_values_normals != 0.5)
                        near_zero_pixels_indices = np.where(near_zero_mask.all(axis=2))
                        dataset_values_normals = (dataset_values_normals * 255).astype(np.uint8)

                        # temp_imageの生成
                        temp_image = np.copy(dataset_values_normals)
                        for y, x in zip(*zero_pixels_indices):
                            temp_image[y, x] = dataset_values_colors[y, x]

                        # composite_imageの生成
                        composite_image = np.copy(temp_image)
                        for y, x in zip(*near_zero_pixels_indices):
                            composite_image[y, x] = 0.8 * dataset_values_colors[y, x] + 0.2 * dataset_values_normals[y, x]

                        image_composite = Image.fromarray(composite_image)

                        # カラー画像とcomposite画像を横に結合
                        combined_image_width = image_colors.width + image_composite.width
                        combined_image_height = max(image_colors.height, image_composite.height)
                        combined_image = Image.new('RGB', (combined_image_width, combined_image_height))
                        combined_image.paste(image_colors, (0, 0))
                        combined_image.paste(image_composite, (image_colors.width, 0))

                        # 結合した画像を保存
                        output_filename = f"{hdri_dir}_{j:04d}.png"
                        output_filepath = os.path.join(output_image_dir, output_filename)
                        combined_image.save(output_filepath)


# メイン関数
def main():
    print("test_final")
    # BlenderProcを初期化
    bproc.init()
    bproc.renderer.set_render_devices(desired_gpu_device_type="CUDA")

    output_rendering_dir =r"C:\Users\kojik\program\source_test22\make_dateset\output_rendering\test"
    os.makedirs(output_rendering_dir, exist_ok=True)

    num_shots = 1000  # カメラのショット数
    input_hdri_path = r"C:\Users\kojik\program\source_test22\make_dateset\hdri_data\test"
    set_scene(input_hdri_path, output_rendering_dir, num_shots)

    output_image_dir =r"C:\Users\kojik\program\source_test22\make_dateset\output_image\test"
    os.makedirs(output_image_dir, exist_ok=True)
    convert_hdf5_to_png(output_rendering_dir, output_image_dir, num_shots)

if __name__ == "__main__":
    main()
