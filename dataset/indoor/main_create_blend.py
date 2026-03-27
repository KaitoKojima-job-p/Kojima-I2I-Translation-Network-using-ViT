import blenderproc as bproc
import bpy
import os
import glob

def set_init_scene(obj_path, hdr_path):
    
    def set_light_bounce_visibility(context):
        scn = context.scene
        # サイクルレンダーのMax Bounce設定はレンダープロパティで行う
        bpy.context.scene.cycles.max_bounces = 12                   # 最大バウンス数（全体）
        bpy.context.scene.cycles.diffuse_bounces = 4                # 拡散反射バウンス
        bpy.context.scene.cycles.glossy_bounces = 4                 # 鏡面反射バウンス
        bpy.context.scene.cycles.transmission_bounces = 12          # 透過バウンス
        bpy.context.scene.cycles.volume_bounces = 2                  # ボリューム散乱バウンス
        bpy.context.scene.cycles.transparent_max_bounces = 8        # 透明シェーダーの最大バウンス

        # ワールドサイクル用可視性を設定
        scn.world.cycles_visibility.camera = True
        scn.world.cycles_visibility.diffuse = True
        scn.world.cycles_visibility.glossy = True
        scn.world.cycles_visibility.transmission = True
        scn.world.cycles_visibility.scatter = True

    c = bpy.context
    scn = c.scene

    set_light_bounce_visibility(context=c)

    print("scene.cycles.device:", scn.cycles.device)    

    # ワールド用ノード構築
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    # オブジェクトロード
    _ = bproc.loader.load_obj(obj_path)

    for obj in bpy.data.objects:
        if obj.name == 'stanford-bunny':
            bunny = obj
            scale = 5.0
            bunny.scale = (scale, scale, scale)
            bunny.location = (2, 10, 0.5)

            # 滑らかにshading
            bpy.context.view_layer.objects.active = bunny
            obj.select_set(True)
            bpy.ops.object.shade_smooth()
            obj.select_set(False)


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
    node_environment = tree_nodes.new(type='ShaderNodeTexEnvironment')
    node_environment.location = (-300, 0)
    node_mapping = tree_nodes.new(type='ShaderNodeMapping')
    node_mapping.location = (-600, 0)
    node_texcoord = tree_nodes.new(type='ShaderNodeTexCoord')
    node_texcoord.location = (-900, 0)
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = (200, 0)

    links = node_tree.links
    links.new(node_texcoord.outputs['Generated'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_environment.inputs['Vector'])
    links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    # 環境画像ロード
    node_environment.image = bpy.data.images.load(hdr_path)

    bproc.camera.set_resolution(256, 256)

#    for obj in bpy.data.objects:
#        obj.hide_render = False


def render_scene(json_file, output_rendering_dir):
    json_base_name = os.path.splitext(os.path.basename(json_file))
    blend_save_path = os.path.join(output_rendering_dir, f"{json_base_name[0]}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)
    print(f"Blendファイルを保存しました: {blend_save_path}")


def set_front(front, future_folder, front_3D_texture_path):
    if not os.path.exists(front) or not os.path.exists(future_folder):
        raise Exception("One of the two folders does not exist!")
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    loaded_objects = bproc.loader.load_front3d(
        json_path=front,
        future_model_path=future_folder,
        front_3D_texture_path=front_3D_texture_path,
        label_mapping=mapping
    )


def delete_except_camera_bunny():
    # 残したいオブジェクト名リスト（必要に応じて調整）
    keep_names = ["Camera", "stanford-bunny"]
    # すべてのオブジェクトを走査
    for obj in bpy.data.objects:
        if obj.name not in keep_names:
            bpy.data.objects.remove(obj, do_unlink=True)


def create_blend(json_files, output_rendering_dir):
    # Example paths
    obj_path = r"C:\Users\kojik\code\program\source_test29\make_dataset\object_data\stanford-bunny.obj"
    hdr_path = r"C:\Users\kojik\code\program\source_test29\make_dataset\hdri_data\test_kakutei\bergen_4k.hdr"
    set_init_scene(obj_path, hdr_path)
    # 各jsonファイル処理
    for json_file in json_files:
        delete_except_camera_bunny()
        front = json_file
        future_folder = r"D:\programD\datasets\3dfront\3D-FUTURE-model"
        front_3D_texture_path = r"D:\programD\datasets\3dfront\3D-FRONT-texture"
        set_front(front, future_folder, front_3D_texture_path)
        render_scene(front, output_rendering_dir)


def main():
    print("test_final")
    bproc.init()
    bproc.renderer.set_render_devices(desired_gpu_device_type="CUDA")
    output_rendering_dir = r"C:\Users\kojik\code\program\source_test29\3D-front\blends\test_zz_temp"
    os.makedirs(output_rendering_dir, exist_ok=True)
    # jsonファイルリスト例
    json_dir = r"C:\Users\kojik\code\program\source_test29\3D-front\jsons\test_zz"
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    create_blend(json_files, output_rendering_dir)

if __name__ == "__main__":
    main()
