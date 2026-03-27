import blenderproc as bproc
import bpy
import os
import argparse
import numpy as np
from PIL import Image
import h5py
from mathutils import Vector, Matrix, Euler, Quaternion
import sys
import math
import csv
import shutil
import random


# コマンドライン引数パース設定を関数外で定義
parser = argparse.ArgumentParser(description="CSVファイルを読み込み複数レンダリング")
parser.add_argument("input_csv", type=str, help="Blendファイルパスと対象オブジェクト名のCSVファイルパス")
args = parser.parse_args()


def place_above(ref_name, target_name):
    """
    ref_nameで指定したオブジェクトのトップの上に、
    target_nameのオブジェクトの最下端が乗るように配置する関数。

    Parameters:
        ref_name (str): 土台となる参照オブジェクト名
        target_name (str): 配置対象のオブジェクト名
        offset (float): 参照物の上端からの高さオフセット（メートル）

    Returns:
        tuple: 配置した位置の (X, Y, Z)
    """

    def get_world_bbox(obj):
        return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    def get_top_z(obj_name):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"オブジェクト '{obj_name}' が見つかりません")
        return max(corner.z for corner in get_world_bbox(obj))

    def get_bottom_z(obj_name):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"オブジェクト '{obj_name}' が見つかりません")
        return min(corner.z for corner in get_world_bbox(obj))

    def get_bbox_center_xy(obj_name):
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"オブジェクト '{obj_name}' が見つかりません")
        corners = get_world_bbox(obj)
        center = sum(corners, Vector()) / len(corners)
        return center.x, center.y
        
    # 操作したいオブジェクトをアクティブにして選択状態にする
    target = bpy.data.objects.get("stanford-bunny")
    if target is None:
        print("オブジェクト 'stanford-bunny' が見つかりません")
    else:
        bpy.context.view_layer.objects.active = target
        target.select_set(True)

        # 原点をバウンディングボックスの中心（質量中心）に設定
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
        bpy.context.view_layer.update()

    ref_top_z = get_top_z(ref_name)
    target_bottom_z = get_bottom_z(target_name)
    cx, cy = get_bbox_center_xy(ref_name)

    target = bpy.data.objects.get(target_name)
    if target is None:
        raise ValueError(f"オブジェクト '{target_name}' が見つかりません")

    offset = target_bottom_z - ref_top_z
    new_z = target.location.z - offset

    return (cx, cy, new_z)

#set_camera_with_visibility_checkのrランダムバージョン
"""
def set_camera_with_visibility_check_(target_coords, target_obj, camera_obj, max_trials=100, theta_phi_trial=100, visible_bbox_corners=7):
    for t_trial in range(theta_phi_trial):
        theta = np.random.uniform(0, 0.5 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        for r_trial in range(max_trials):
            r = np.random.uniform(3, 5)
            x = r * np.sin(theta) * np.cos(phi) + target_coords[0]
            y = r * np.sin(theta) * np.sin(phi) + target_coords[1]
            z = r * np.cos(theta) + target_coords[2]
            rx = theta
            ry = 0
            rz = phi + np.pi / 2
            bbox_corners = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]

            cam_pose = bproc.math.build_transformation_mat((x, y, z), (rx, ry, rz))
            camera_obj.matrix_world = Matrix(cam_pose)

            visible_count = 0
            for i, corner in enumerate(bbox_corners):
                direction = (corner - camera_obj.location).normalized()
                distance = (corner - camera_obj.location).length
                result, location, normal, index, hit_obj, matrix = bpy.context.scene.ray_cast(
                    bpy.context.evaluated_depsgraph_get(), camera_obj.location, direction, distance=distance)
                if result:
                    if hit_obj == target_obj:
                        visible_count += 1
                else:
                    visible_count += 1

            if visible_count >= visible_bbox_corners:
                print(f"theta/phi試行:{t_trial+1} r試行:{r_trial+1} 見えている頂点数: {visible_count} r:{r}")
                print(f"決定theta: {theta} phi: {phi}")
                return (x, y, z), (rx, ry, rz), r

    raise RuntimeError("条件を満たすカメラ位置が見つかりませんでした")
"""

# グローバル変数： (index, save_prefix, t_trialに対する平均r_trial回数) を保持
t_trial_records = {}
def set_camera_with_visibility_check(target_coords, target_obj, camera_obj, save_prefix,
                                    current_shot=None, total_shots=None,
                                    obj_rotation_retry_count=None, max_obj_rotation_retries=None,
                                    max_trials=50, theta_phi_trials=100,
                                    visible_threshold=0.99, sample_num=200):
    max_r = 3.0
    min_r = 1.5
    step = (max_r - min_r) / max_trials

    mesh = target_obj.data
    vertices_world_all = [target_obj.matrix_world @ v.co for v in mesh.vertices]
    vertices_world = random.sample(vertices_world_all, sample_num)

    for t_trial in range(theta_phi_trials):
        theta = np.random.uniform(0, 0.5 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        r_values = np.arange(max_r, min_r - step, -step)
        for r_trial, r in enumerate(r_values):
            progress_msg = f"[{save_prefix}] "
            if current_shot is not None and total_shots is not None:
                progress_msg += f"{current_shot}/{total_shots}枚目 "
            if obj_rotation_retry_count is not None and max_obj_rotation_retries is not None:
                progress_msg += f"obj回転試行 {obj_rotation_retry_count}/{max_obj_rotation_retries} "
            progress_msg += f"t_trial {t_trial + 1}/{theta_phi_trials}"

            print(f"\r{progress_msg}    ", end="", flush=True)

            x = r * np.sin(theta) * np.cos(phi) + target_coords[0]
            y = r * np.sin(theta) * np.sin(phi) + target_coords[1]
            z = r * np.cos(theta) + target_coords[2]
            rx, ry, rz = theta, 0, phi + np.pi / 2

            cam_pose = bproc.math.build_transformation_mat((x, y, z), (rx, ry, rz))
            camera_obj.matrix_world = Matrix(cam_pose)
            bpy.context.view_layer.update()

            bbox_corners = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]

            bbox_clear = True
            for point in bbox_corners:
                direction = (point - camera_obj.location).normalized()
                distance = (point - camera_obj.location).length

                result, location, normal, index, hit_obj, matrix = bpy.context.scene.ray_cast(
                    bpy.context.evaluated_depsgraph_get(), camera_obj.location, direction, distance=distance)

                if result and hit_obj != target_obj:
                    bbox_clear = False
                    break

            if not bbox_clear:
                continue

            visible_count = 0
            for point in vertices_world:
                direction = (point - camera_obj.location).normalized()
                distance = (point - camera_obj.location).length

                result, location, normal, index, hit_obj, matrix = bpy.context.scene.ray_cast(
                    bpy.context.evaluated_depsgraph_get(), camera_obj.location, direction, distance=distance)

                if result and hit_obj == target_obj:
                    visible_count += 1
                elif not result:
                    visible_count += 1

            visibility_ratio = visible_count / len(vertices_world)
            if visibility_ratio >= visible_threshold:
                return  # 条件達成。呼び出し元に戻る

    raise RuntimeError("条件を満たすカメラ位置が見つかりませんでした")


# 外部で平均t_trial回数を集計し表示
def print_average_t_trials():
    print("=== save_prefixごとの平均t_trial回数（角度方向試行回数） ===")
    for prefix, t_trials_list in t_trial_records.items():
        avg_t = sum(t_trials_list) / len(t_trials_list) if len(t_trials_list) > 0 else 0
        print(f"{prefix}: 平均 t_trial 回数 = {avg_t:.2f}")


def render_scene(num_shots, target_name, hdf5_output_dir, save_prefix, max_obj_rotation_retries=50):
    for obj in bpy.data.objects:
        obj.hide_render = False
    
    def get_world_mapping_node():
        world = bpy.context.scene.world
        if world is None or not world.use_nodes:
            raise Exception("ワールドノードが有効になっていません")

        node_tree = world.node_tree
        for node in node_tree.nodes:
            if node.type == 'MAPPING':
                return node
        raise Exception("Mappingノードが見つかりません")
    
    # オブジェクト取得
    bunny = bpy.data.objects.get(target_name)
    if bunny is None:
        raise ValueError(f"オブジェクト '{target_name}' が見つかりません")

    # ワールドのMappingノード取得
    node_mapping = get_world_mapping_node()

    # カメラ設定
    bproc.camera.set_resolution(256, 256)
    camera_obj = bpy.context.scene.camera
    focal_length= 30
    camera_obj.data.lens = focal_length

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0

    for k in range(num_shots):
        bunny = bpy.data.objects.get(target_name)
        bunny_euler = bunny.rotation_euler.copy()

        # 環境マップをランダム回転（Z軸）
        env_rot_z = np.random.uniform(0, 2 * np.pi)
        node_mapping.inputs['Rotation'].default_value[2] = env_rot_z
        node_mapping.inputs['Rotation'].keyframe_insert(data_path="default_value", frame=k) 


        obj_rotation_retries = 0
        while obj_rotation_retries < max_obj_rotation_retries:
            try:
                obj_rotation_retries += 1

                # オブジェクトをランダム回転
                z_rot_offset = np.random.uniform(0, np.pi / 2)
                bunny.rotation_euler = (bunny_euler[0], bunny_euler[1], bunny_euler[2] + z_rot_offset)
                bunny.keyframe_insert(data_path="rotation_euler", frame=k)

                # カメラ配置試行
                set_camera_with_visibility_check(
                    bunny.location, bunny, bpy.context.scene.camera, save_prefix,
                    current_shot=k+1, total_shots=num_shots,
                    obj_rotation_retry_count=obj_rotation_retries,
                    max_obj_rotation_retries=max_obj_rotation_retries)

                # 成功ならループ脱出
                break

            except RuntimeError as e:
                if obj_rotation_retries >= max_obj_rotation_retries:
                    # 最大回数達成で条件未達表示し例外再スロー
                    fail_msg = f"\n[{save_prefix}] {k+1}/{num_shots} 枚目 条件未達 (obj回転試行 {obj_rotation_retries}/{max_obj_rotation_retries})"
                    print(fail_msg)
                    raise RuntimeError(f"{max_obj_rotation_retries}回のobj回転試行とt_trial試行で失敗しました: {e}")
                else:
                    # 続行。obj回転試行回数が増えた分だけ上書き表示される
                    continue

        # カメラ設定が成功した場合のみレンダー登録
        camera_obj.rotation_mode = 'QUATERNION'
        camera_obj.keyframe_insert(data_path='location', frame=k)
        camera_obj.keyframe_insert(data_path='rotation_euler', frame=k)

        forward_direction = camera_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
        forward_direction.normalize()

        random_rotation_rad = np.random.uniform(-np.pi / 6, np.pi / 6)
        rotation_quat = Quaternion(forward_direction, random_rotation_rad)
        new_rotation = rotation_quat @ camera_obj.rotation_quaternion
        camera_obj.rotation_quaternion = new_rotation
        camera_obj.keyframe_insert(data_path='rotation_quaternion', frame=k)

        bpy.context.scene.frame_end += 1

    # 法線出力を有効化してレンダリング
    bproc.renderer.enable_normals_output()

    # 正解画像作成
    data = bproc.renderer.render()
    c_dir = os.path.join(hdf5_output_dir, "color")
    os.makedirs(c_dir, exist_ok=True)
    bproc.writer.write_hdf5(c_dir, data)

    # 法線画像作成
    # bunnyなし家具あり
    for obj in bpy.data.objects:
        # 名前が "stanford-bunny" なら非表示に設定
        if obj.name == "stanford-bunny":
            obj.hide_render = True
    #カラーで作成
    data = bproc.renderer.render()
    cwo_dir = os.path.join(hdf5_output_dir, "color_wo_bunny")
    bproc.writer.write_hdf5(cwo_dir, data)

    # bunnyあり家具なし
    for obj in bpy.data.objects:
        # 名前が "stanford-bunny" 以外なら非表示に設定
        if obj.name != "stanford-bunny":
            obj.hide_render = True
        else:
            # 念のためバニーはレンダー可視に
            obj.hide_render = False
    
    data = bproc.renderer.render()
    n_dir = os.path.join(hdf5_output_dir, "normal")
    bproc.writer.write_hdf5(n_dir, data)

    bpy.context.scene.frame_start = bpy.context.scene.frame_end


def convert_hdf5_to_png(input_dir, output_image_dir, num_shots, save_prefix):
    for i in range(num_shots):
        input_file_c = os.path.join(input_dir, "color", f"{i}.hdf5")
        input_file_cwo = os.path.join(input_dir, "color_wo_bunny", f"{i}.hdf5")
        input_file_n = os.path.join(input_dir, "normal", f"{i}.hdf5")
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
                    output_filename = f"{save_prefix}_{i:04d}.png"
                    output_filepath = os.path.join(output_image_dir, output_filename)
                    combined_image.save(output_filepath)


def main():
    bproc.init()

    input_csv_path = args.input_csv

    output_hdf5 = r"C:\Users\kojik\code\program\source_test29\3D-front\output_hdf5"
    os.makedirs(output_hdf5, exist_ok=True)

    output_image = r"C:\Users\kojik\code\program\source_test29\3D-front\output_image"
    os.makedirs(output_image, exist_ok=True)

    #一つの地点における枚数
    num_shots = 1000

    # カメラセット失敗家具のリスト
    failed_furniture = [] 

    with open(input_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            blend_path = row['blend_path']
            obj_name = row['object_name']
            target_name = "stanford-bunny"

            # blendファイルを開く
            bpy.ops.wm.open_mainfile(filepath=blend_path)

            # objスケーリング等初期設定
            bunny = bpy.data.objects.get(target_name)
            bunny.scale = (7.0, 7.0, 7.0)  # 例、必要に応じて調整
            bpy.context.view_layer.update()

            new_coords = place_above(ref_name=obj_name, target_name=target_name)
            bunny.location = new_coords
            bpy.context.view_layer.update()

            # 保存ファイル名 prefix: blendファイル名の8文字まで＋obj名
            blend_name = os.path.splitext(os.path.basename(blend_path))[0][:8]
            save_prefix = f"{blend_name}_{obj_name}"

            # 保存ディレクトリ（例）をblendごとに分ける
            each_output_hdf5 = os.path.join(output_hdf5, save_prefix)
            os.makedirs(each_output_hdf5, exist_ok=True)

            each_output_image = os.path.join(output_image, save_prefix)
            os.makedirs(each_output_image, exist_ok=True)

            try:
                render_scene(num_shots, target_name, each_output_hdf5, save_prefix)

            except RuntimeError as e:
                # 何らかのエラーをキャッチして家具情報と位置を記録
                failed_furniture.append((save_prefix))
                print(f"警告：{obj_name}の処理に失敗しました。原因: {str(e)}")

                # 生成してしまった空ディレクトリを削除（あれば）
                if os.path.isdir(each_output_hdf5):
                    try:
                        shutil.rmtree(each_output_hdf5)
                        print(f"削除しました: {each_output_hdf5}")
                    except Exception as del_err:
                        print(f"ディレクトリ削除失敗: {each_output_hdf5} 理由: {str(del_err)}")

                if os.path.isdir(each_output_image):
                    try:
                        shutil.rmtree(each_output_image)
                        print(f"削除しました: {each_output_image}")
                    except Exception as del_err:
                        print(f"ディレクトリ削除失敗: {each_output_image} 理由: {str(del_err)}")
                        
                # 次の家具へ継続
                continue

            convert_hdf5_to_png(each_output_hdf5, each_output_image, num_shots, save_prefix)

    print_average_t_trials()

    # ループ終了後に失敗した家具一覧を出力
    if failed_furniture:
        print("\n### カメラ位置設定に失敗した家具一覧 ###")
        for name in failed_furniture:
            print(f"家具名: {name}")



if __name__ == "__main__":
    main()


