import blenderproc as bproc
import bpy
import numpy as np
import os
import sys
import h5py
from PIL import Image
import math
import glob
from mathutils import Vector, Matrix, Euler, Quaternion

import argparse

import bpy
from mathutils import Vector


def place_above(ref_name, target_name, offset=0.0):
    """
    ref_name で指定したオブジェクトのトップの上に、
    target_name のオブジェクトの最下端が乗るように配置する関数。
    
    Parameters:
        ref_name (str): 土台となる参照オブジェクト名
        target_name (str): 配置対象のオブジェクト名
        offset (float): 参照物の上端からの高さオフセット（メートル）
    
    Returns:
        tuple: 配置した位置の (X, Y, Z)
    """

    import bpy
    from mathutils import Vector

    # オブジェクトのローカル境界ボックス（bound_box）の8頂点を
    # ワールド座標に変換してリストで返す関数。
    def get_world_bbox(obj):
        return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    # 指定したオブジェクトの境界ボックスの中で最も高い Z 座標を取得する関数
    def get_top_z(obj_name):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"オブジェクト '{obj_name}' が見つかりません")
        world_corners = get_world_bbox(obj)
        max_z = max(corner.z for corner in world_corners)
        return max_z

    # 指定したオブジェクトの境界ボックスの中で最も低い Z 座標を取得する関数
    def get_bottom_z(obj_name):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"オブジェクト '{obj_name}' が見つかりません")
        world_corners = get_world_bbox(obj)
        min_z = min(corner.z for corner in world_corners)
        return min_z

    # 指定したオブジェクトの境界ボックス中心の X,Y 座標をワールド座標で返す関数
    def get_bbox_center_xy(obj_name):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"オブジェクト '{obj_name}' が見つかりません")
        world_corners = get_world_bbox(obj)
        avg = Vector((0.0, 0.0, 0.0))
        for c in world_corners:
            avg += c
        center = avg / len(world_corners)
        return center.x, center.y
    
    # 参照オブジェクトの最高 Z 座標
    ref_top_z = get_top_z(ref_name)

    # 配置対象オブジェクトの最下端 Z 座標
    target_bottom_z = get_bottom_z(target_name)

    # 参照オブジェクトの X,Y 中心座標を取得
    cx, cy = get_bbox_center_xy(ref_name)

    # targetオブジェクトを取得
    target = bpy.data.objects.get(target_name)
    if target is None:
        raise ValueError(f"オブジェクト '{target_name}' が見つかりません")

    # Z位置は、refのトップ + offset に targetの最下端が来るように調整
    new_z = (ref_top_z + offset) - target_bottom_z
    
    # targetの位置を更新（X,Yはref中心に揃える）
    target.location = (cx, cy, new_z)

    # 配置後のtargetの最下端Z座標を再計算
    target_bottom_z_after = get_bottom_z(target_name)
    ref_top_z_offset = ref_top_z + offset
    print("---- Z位置確認 ----")
    print(f"refオブジェクト(‘{ref_name}’)の最上端Z + offset: {ref_top_z_offset:.6f}")
    print(f"配置target(‘{target_name}’)の最下端Z: {target_bottom_z_after:.6f}")
    print(f"差分: {abs(ref_top_z_offset - target_bottom_z_after):.6e}")

    # 必要に応じてキーフレーム挿入も可能（必要ならコメント外す）
    # target.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

    # 配置座標を返す
    return (cx, cy, new_z)


def set_camera(coords):
    r = np.random.uniform(0.5, 0.75)
    theta = np.random.uniform(0.5/2, 0.9/2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)

    x = r * np.sin(theta) * np.cos(phi) + coords[0]
    y = r * np.sin(theta) * np.sin(phi) + coords[1]
    z = r * np.cos(theta) + coords[2]

    rx = theta
    ry = 0
    rz = phi + (np.pi / 2)

    return (x, y, z), (rx, ry, rz), r
 

def set_scene(input_hdri_path, output_dir, num_shots):
    front = r"D:\programD\datasets\3dfront\3D-FRONT\5c1179cf-f40c-456b-ad4f-b3f70c7c5f02.json"  # 3D frontファイルの固定パス（例）
    future_folder = r"D:\programD\datasets\3dfront\3D-FUTURE-model"  # Future Modelフォルダの固定パス（例）
    front_3D_texture_path = r"D:\programD\datasets\3dfront\3D-FRONT-texture"  # FRONT textureフォルダの固定パス（例）

    if not os.path.exists(front) or not os.path.exists(future_folder):
        raise Exception("One of the two folders does not exist!")

    # 以降は args.front の代わりに front を使うなど修正
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # set the light bounces
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                    transmission_bounces=200, transparent_max_bounces=200)

    c = bpy.context
    scn = c.scene

    print("scene.cycles.device:", scn.cycles.device)

    c.scene.world.cycles_visibility.camera = True
    c.scene.world.cycles_visibility.Diffuse = True
    c.scene.world.cycles_visibility.Glossy = True
    c.scene.world.cycles_visibility.Transmission = True
    c.scene.world.cycles_visibility.Volume_Scatter = True

    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    hdri_directory_path = input_hdri_path
    hdri_files = [f for f in os.listdir(hdri_directory_path) if f.endswith('.hdr')]

    # 例：load_front3d
    loaded_objects = bproc.loader.load_front3d(
        json_path=front,
        future_model_path=future_folder,
        front_3D_texture_path=front_3D_texture_path,
        label_mapping=mapping
    )
    
    # load the objects into the scene
    obj_path = r"C:\Users\kojik\code\program\source_test29\make_dataset\object_data\stanford-bunny.obj"
    objs = bproc.loader.load_obj(obj_path)
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=640,       # 経度方向の分割数
        ring_count=320,     # 緯度方向の分割数
        radius=1,        # 半径
        enter_editmode=False,
        align='WORLD',
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0)
    )
    # 直近で生成されたオブジェクトを取得して名前を変更
    sphere = bpy.context.active_object
    sphere.name = "stanford-bunny"
    bpy.ops.object.origin_set()
    """

    for obj in bpy.data.objects:
        if (obj.name == 'stanford-bunny'):
            bunny = obj
            scale = 1.0

            bunny.scale = (scale, scale, scale)
            e = obj.rotation_euler.copy()

    coords = place_above("table.004", "stanford-bunny")
    print(f"配置座標: X={coords[0]:.3f}, Y={coords[1]:.3f}, Z={coords[2]:.3f}")

    metallic = 1.0
    specular = 1.0
    roughness = 0.05

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
    node_mapping = tree_nodes.new(type='ShaderNodeMapping')
    node_mapping.location = -600, 0
    node_texcoord = tree_nodes.new(type='ShaderNodeTexCoord')
    node_texcoord.location = -900, 0
    links = node_tree.links
    links.new(node_texcoord.outputs['Generated'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_environment.inputs['Vector'])
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
    node_output.location = 200,0
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
            z_rotation = np.random.uniform(0, 2 * np.pi)
            node_mapping.inputs['Rotation'].default_value[2] = z_rotation

            z_rotation = np.random.uniform(0, np.pi/2)
            bunny.rotation_euler = (e[0], e[1], e[2]+z_rotation)
            bunny.keyframe_insert(data_path="rotation_euler", frame=k)
            #print(bunny.rotation_euler)

            bunny.location = (coords[0], coords[1], coords[2])
            bunny.keyframe_insert(data_path="location", frame=k)
        
            camera_location, camera_rotation, r = set_camera(coords)

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

        # 正解画像作成
        data = bproc.renderer.render()
        c_dir = os.path.join(save_dir, "color")
        bproc.writer.write_hdf5(c_dir, data)

        # 法線画像作成
        # bunnyなし家具あり
        for obj in bpy.data.objects:
            # 名前が "stanford-bunny" なら非表示に設定
            if obj.name == "stanford-bunny":
                obj.hide_render = True
        #カラーで作成
        data = bproc.renderer.render()
        cwo_dir = os.path.join(save_dir, "color_wo_bunny")
        bproc.writer.write_hdf5(cwo_dir, data)

        # bunnyあり家具なし
        for obj in bpy.data.objects:
            # 名前が "stanford-bunny" 以外なら非表示に設定
            if obj.name != "stanford-bunny":
                obj.hide_render = True
            else:
                # 念のためバニーはレンダー可視に
                obj.hide_render = False
        # 法線出力を有効化してレンダリング
        bproc.renderer.enable_normals_output()
        data = bproc.renderer.render()
        n_dir = os.path.join(save_dir, "normal")
        bproc.writer.write_hdf5(n_dir, data)

        bpy.context.scene.frame_start = bpy.context.scene.frame_end

    output_rendering_dir = r"C:\Users\kojik\code\program\source_test29\3D-front\blends\test_zz_temp"
    blend_save_path = os.path.join(output_rendering_dir, "scene_saved.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)
    print(f"Blendファイルを保存しました: {blend_save_path}")

    bproc.clean_up


def convert_hdf5_to_png(input_dir, output_image_dir, num_shots):
    # 入力ディレクトリ内のサブディレクトリを処理
    for i, hdri_dir in enumerate(os.listdir(input_dir)):  # enumerate でインデックスを取得
        hdri_path = os.path.join(input_dir, hdri_dir)
        if os.path.isdir(hdri_path):  # ディレクトリかどうかを確認
            for j in range(num_shots):
                input_file_c = os.path.join(hdri_path, "color", f"{j}.hdf5")
                input_file_cwo = os.path.join(hdri_path, "color_wo_bunny", f"{j}.hdf5")
                input_file_n = os.path.join(hdri_path, "normal", f"{j}.hdf5")
                with h5py.File(input_file_c, 'r') as fc:
                    with h5py.File(input_file_cwo, 'r') as fcwo:
                        with h5py.File(input_file_n, 'r') as fn:
                            # カラー画像の読み込み
                            dataset_values_colors = np.array(fc['colors'])
                            image_colors = Image.fromarray(dataset_values_colors)

                            # bunnyなしカラー画像の読み込み
                            dataset_values_colors_wo_bunny = np.array(fcwo['colors'])

                            # 法線画像の読み込みと処理
                            dataset_values_normals = np.array(fn['normals'])
                            threshold = 0.20
                            zero_mask = (dataset_values_normals == 0.5)
                            zero_pixels_indices = np.where(zero_mask.all(axis=2))
                            near_zero_mask = (dataset_values_normals >= 0.5 - threshold) & (dataset_values_normals <= 0.5 + threshold) & (dataset_values_normals != 0.5)
                            near_zero_pixels_indices = np.where(near_zero_mask.all(axis=2))
                            dataset_values_normals = (dataset_values_normals * 255).astype(np.uint8)

                            # temp_imageの生成
                            temp_image = np.copy(dataset_values_normals)
                            for y, x in zip(*zero_pixels_indices):
                                temp_image[y, x] = dataset_values_colors_wo_bunny[y, x]

                            # composite_imageの生成
                            composite_image = np.copy(temp_image)
                            for y, x in zip(*near_zero_pixels_indices):
                                composite_image[y, x] = 0.8 * dataset_values_colors_wo_bunny[y, x] + 0.2 * dataset_values_normals[y, x]
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

    output_rendering_dir = r"C:\Users\kojik\code\program\source_test29\make_dataset\output_rendering\test_3dfront"
    os.makedirs(output_rendering_dir, exist_ok=True)

    num_shots = 1  # カメラのショット数
    input_hdri_path =  r"C:\Users\kojik\code\program\source_test29\make_dataset\hdri_data\test_3dfront"
    set_scene(input_hdri_path, output_rendering_dir, num_shots)

    output_image_dir = r"C:\Users\kojik\code\program\source_test29\make_dataset\output_image\test_3dfront_____"
    os.makedirs(output_image_dir, exist_ok=True)
    #convert_hdf5_to_png(output_rendering_dir, output_image_dir, num_shots)

if __name__ == "__main__":
    main()
