import logging
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import bpy
import os
import subprocess
# --- Some configuration values
SPAWN_REGION = [(-7, -7, 1), (7, 7, 7)]
VELOCITY_RANGE = [(-2., -2., 0.), (2., 2., 0.)]
CLEVR_OBJECTS = ("cube", "cylinder", "sphere")
KUBASIC_OBJECTS = ("cube", "cylinder", "sphere", "cone", "torus", "gear",
                   "torus_knot", "sponge", "spot", "teapot", "suzanne")
# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")
parser.add_argument("--min_num_objects", type=int, default=3,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=8,
                    help="maximum number of objects")
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--camera", choices=["fixed_random", "linear_movement"], default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=True, frame_end=120, frame_rate=30, resolution=256)
# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")

FLAGS = parser.parse_args()
num_sets = 1 # 要生成的视频组数
for set_index in range(num_sets):

    # --- Common setups & resources
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
    scene.gravity = (0, 0, -9.81)  # 设置重力方向向下
    output_subdir = os.path.join(output_dir, f'obj_{set_index}')
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    # Lights
    logging.info("Adding four (studio) lights to the scene similar to CLEVR...")
    scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
    # Dome
    floor_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)

    dome = kubasic.create(asset_id="dome", name="dome",material=floor_material,
                          friction=FLAGS.floor_friction,
                          restitution=FLAGS.floor_restitution,
                          static=True, background=True)
    floor_material.color = kb.Color.from_name("gray")
    scene.metadata["background"] = "clevr"
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]


    #
    # # 创建斜坡（使用圆锥体）
    # ramp = kubasic.create(
    #     asset_id="cone",
    #     name="ramp",
    #     scale=8.0 # 使用均匀的缩放比例
    # )
    #
    # # 设置旋转和位置
    #
    # ramp.position = (0, 0, 1)  # 设置圆锥体的位置
    # ramp.static = True
    # scene += ramp
    #
    # ramp_blender = ramp.linked_objects[renderer]
    # # 设置物体的旋转，确保绕X轴旋转30度
    # ramp_blender.rotation_mode = 'XYZ'  # 确保使用正确的旋转模式
    # ramp_blender.rotation_euler = (np.radians(120), 0, 0)  # 绕x轴旋转30度
    #
    # # 应用变换以确保旋转在Blender中生效
    # bpy.context.view_layer.objects.active = ramp_blender
    # bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    # # 将对象添加到场景中
    plane = kubasic.create(
        asset_id="cube",
        name="cube",
        scale=8.0  # 使用均匀的缩放比例,

    )

    # 设置旋转和位置

    plane.position = (-2, 0, -1.5)  # 设置圆锥体的位置
    plane.static = True
    scene += plane
    plane.background = True
    plane_blender = plane.linked_objects[renderer]
    # 设置物体的旋转，确保绕X轴旋转30度
    plane_blender.rotation_mode = 'XYZ'  # 确保使用正确的旋转模式
    plane_blender.rotation_euler = (np.radians(120), 0, 0)  # 绕x轴旋转30度

    # 应用变换以确保旋转在Blender中生效
    bpy.context.view_layer.objects.active = plane_blender
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    # Create obj
    # 添加的代码
    def generate_object_properties(existing_objects, rng):
        while True:
            shape_name = rng.choice(CLEVR_OBJECTS)
            size_label, size = kb.randomness.sample_sizes("clevr", rng)
            color_label, random_color = kb.randomness.sample_color("clevr", rng)
            material_name = rng.choice(["metal", "rubber"])

            # 生成新的物体属性
            new_properties = {
                "shape_name": shape_name,
                "size_label": size_label,
                "color_label": color_label,
                "material_name": material_name,
                "random_color": random_color,
                "size": size
            }

            # 检查与已有物体的属性是否有完全重复
            if not any(all(obj[property] == new_properties[property] for property in new_properties) for obj in
                       existing_objects):
                return new_properties

    existing_objects = []  # 存储已生成物体的属性
    # Add random objects
    num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects + 1)
    logging.info("Randomly placing %d objects:", num_objects)
    generated_objects = []
    def is_overlapping(obj, existing_objects, padding=0.5):
        """检查 obj 是否与 existing_objects 中的任意物体重叠，padding 用于确保有一定的安全距离"""
        for existing_obj in existing_objects:
            distance = np.linalg.norm(np.array(obj.position) - np.array(existing_obj.position))
            if distance < (obj.scale + existing_obj.scale) / 2 + padding:
                return True
        return False
    for i in range(num_objects):
        properties = generate_object_properties(existing_objects, rng)

        obj = kubasic.create(
            asset_id=properties["shape_name"],
            scale=properties["size"],
            name=f"{properties['size_label']} {properties['color_label']} {properties['material_name']} {properties['shape_name']}",

        )

        assert isinstance(obj, kb.FileBasedObject)

        # 尝试生成不重叠的位置
        positioned = False
        attempts = 0
        max_attempts = 10  # 限制尝试次数以避免无限循环

        while not positioned and attempts < max_attempts:
            obj.position = rng.uniform(SPAWN_REGION[0], SPAWN_REGION[1])
            if not is_overlapping(obj, generated_objects):  # 检查是否重叠
                positioned = True
            attempts += 1

        if not positioned:
            logging.warning(f"Could not find a non-overlapping position for {obj.name} after {max_attempts} attempts.")
            continue  # 跳过当前物体，如果没有找到合适的位置

        # 设置物体的材料和物理属性
        # ...
        if properties["material_name"] == "metal":
            obj.material = kb.PrincipledBSDFMaterial(color=properties["random_color"], metallic=1.0,
                                                     roughness=0.2, ior=2.5)
            obj.friction = 0.4
            # obj.friction = 0
            obj.restitution = 0.3
            obj.mass =rng.randint(3,6) * properties["size"]**3
        else:  # material_name == "rubber"
            obj.material = kb.PrincipledBSDFMaterial(color=properties["random_color"], metallic=0.,
                                                     ior=1.25, roughness=0.7,
                                                     specular=0.33)
            # obj.friction = 0
            obj.friction = 0.8
            obj.restitution = 0.7
            obj.mass *= rng.randint(1,3) * properties["size"] ** 3

        obj.metadata = {
            "shape": properties["shape_name"].lower(),
            "size": properties["size"],
            "size_label": properties["size_label"],
            "material": properties["material_name"].lower(),
            "color": properties["random_color"].rgb,
            "color_label": properties["color_label"],
        }
        scene.add(obj)
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)

        # initialize velocity randomly but biased towards center
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])

        logging.info("    Added %s at %s", obj.asset_id, obj.position)
        existing_objects.append(properties)  # 存储已生成物体的属性
        generated_objects.append(obj)  # 添加物体到列表

    #摩擦系数
    dome.friction = FLAGS.floor_friction  # 设置摩擦系数
    dome.restitution = FLAGS.floor_restitution  # 设置恢复系数

    #物体位置
    positions = []  # 用于存储每一帧的位置
    orientation = []  # 用于存储每一帧的位置
    output_file = 'object_positions.txt'
    full_path = os.path.join(output_subdir, output_file)
    # --- Camera setup
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
    if FLAGS.camera == "fixed_random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7., outer_radius=9., offset=0.1)
        scene.camera.look_at((0, 0, 0))
        scene.camera.sensor_width = 64  # 增加传感器宽度
        scene.camera.focal_length = 20  # 减小焦距以增加视野范围
    elif FLAGS.camera == "linear_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement))
        for frame in range(FLAGS.frame_end + 2):
            interp = (frame - 1) / (FLAGS.frame_end + 1)
            scene.camera.position = (interp * np.array(camera_start) + (1 - interp) * np.array(camera_end))
            scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)


    # --- Run simulation
    logging.info("Running the simulation ...")
    animation = {}
    collisions = []
    print()
    with open(full_path, 'w') as file:

        for frame_index in range(FLAGS.frame_end):

            animation, collisions = simulator.run(frame_start=frame_index, frame_end=frame_index+1)

            # 遍历场景中的所有物体
            for obj in bpy.context.scene.objects:

                # 获取 Kubric 对象的名称
                kb_obj_name = obj.name
                # 检查物体名称是否在生成的物体列表中
                if kb_obj_name in [gen_obj.name for gen_obj in generated_objects]:
                    positions = obj.location
                    orientation = obj.rotation_quaternion  # 获取四元数表示的方向
                    animation.setdefault(obj.name, []).append((positions, orientation))

                    # 将信息写入文件
                    file.write(f"Frame {frame_index}: Object {obj.name} - Position={positions}, Orientation={orientation}\n")
        for obj in generated_objects:
            mass = obj.mass
            friction = obj.friction
            restitution = obj.restitution
            file.write(f"Object {obj.name} - Mass={mass}, Friction={friction}, Restitution={restitution}\n")

    # --- Rendering
    if FLAGS.save_state:
        logging.info(f"Saving the renderer state to '{output_subdir}/scene_{set_index}.blend'")
        blend_file_path = os.path.join(output_subdir, f"scene_{set_index}.blend")
        renderer.save_state(blend_file_path)

    logging.info("Rendering the scene...")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(visible_foreground_assets, key=lambda asset: np.sum(asset.metadata["visibility"]), reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(data_stack["segmentation"], scene.assets, visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    kb.write_image_dict(data_stack, output_subdir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    metadata_file_path = os.path.join(output_subdir, "metadata.json")
    events_file_path = os.path.join(output_subdir, "events.json")
    kb.write_json(filename=metadata_file_path, data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=events_file_path, data={
        "collisions": kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
    })

    kb.done()

# # 设置图像帧所在目录和输出视频文件的路径
# frames_dir = output_subdir
# output_video = os.path.join(output_subdir,'video.mp4')
#
# # 确保frames_dir路径存在
# if not os.path.exists(frames_dir):
#     raise FileNotFoundError(f"Frames directory '{frames_dir}' not found")
#
# # 查找帧图像文件，假设它们是按顺序命名的，比如 frame0001.png, frame0002.png, ...
# frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
# if not frame_files:
#     raise FileNotFoundError("No frame images found in the specified directory")
#
# # 构建ffmpeg命令
# ffmpeg_cmd = [
#     "ffmpeg",
#     "-framerate", "30",  # 设置帧率，可以根据需要调整
#     "-i", os.path.join(frames_dir, "rgba_%05d.png"),  # 输入文件模式
#     "-c:v", "libx264",  # 使用H.264编码
#     "-pix_fmt", "yuv420p",  # 设置像素格式
#     "-crf", "18",  # 设置恒定质量因子，数值越小质量越高
#     output_video
# ]
#
# # 执行ffmpeg命令
# subprocess.run(ffmpeg_cmd, check=True)
