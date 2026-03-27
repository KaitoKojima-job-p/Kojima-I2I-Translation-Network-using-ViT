import blenderproc as bproc
import bpy
import os

def save_all_jsons_as_blend(json_dir, output_dir):
    # json_dir内のjsonファイル一覧を取得
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        raise Exception("jsonディレクトリにjsonファイルが見つかりません")

    # 共通パス
    future_folder = r"D:\programD\datasets\3dfront\3D-FUTURE-model"
    front_3D_texture_path = r"D:\programD\datasets\3dfront\3D-FRONT-texture"
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    for json_name in json_files:
        front = os.path.join(json_dir, json_name)
        print(f"Processing: {front}")

        if not os.path.exists(front):
            print(f"ファイルが見つかりません: {front}")
            continue

        # シーン初期化
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # CyclesレンダーエンジンとCUDA明示
        bpy.context.scene.render.engine = 'CYCLES'

        # 必ず先にタイプ指定し、初期化する
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

        # デバイスリストの初期化
        bpy.context.preferences.addons["cycles"].preferences.get_devices()

        # ここでデバイス列挙し、有効化
        for device in bpy.context.preferences.addons["cycles"].preferences.devices:
            if device.type == 'CUDA' or device.type == 'OPTIX':
                device.use = True

        bpy.context.scene.cycles.device = 'GPU'


        # シーン読み込み
        loaded_objects = bproc.loader.load_front3d(
            json_path=front,
            future_model_path=future_folder,
            front_3D_texture_path=front_3D_texture_path,
            label_mapping=mapping
        )

        # レンダー設定
        bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                         transmission_bounces=200, transparent_max_bounces=200)

        blend_save_name = os.path.splitext(json_name)[0] + ".blend"
        blend_save_path = os.path.join(output_dir, blend_save_name)

        bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)
        print(f"Blendファイルを保存しました: {blend_save_path}")

def main():
    bproc.init()
    json_dir = r"C:\Users\kojik\code\program\source_test29\3D-front\jsons\test_zz"
    output_dir = r"C:\Users\kojik\code\program\source_test29\3D-front\blends\test_zz"
    os.makedirs(output_dir, exist_ok=True)
    save_all_jsons_as_blend(json_dir, output_dir)

if __name__ == "__main__":
    main()
