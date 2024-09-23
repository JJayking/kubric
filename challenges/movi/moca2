import logging
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import bpy
import os
import random
import bmesh
from scipy.spatial.transform import Rotation as R
# --- Some configuration values
SPAWN_REGION = [(-5, -7, 0), (5.6, 7.2, 6)]
SPAWN_REGIONx = [-5,5]
SPAWN_REGIONy = [-7,7]
VELOCITY_RANGE = [(-2., -2., 0), (2., 2., 0)]
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
parser.set_defaults(save_state=True, frame_end=60, frame_rate=30, resolution=256)
# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")

FLAGS = parser.parse_args()


# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
scene.gravity = (0, 0, -9.81)  # 设置重力方向向下
COULOMB_CONSTANT = 4.99e8  # 库仑常数


def compute_electromagnetic_force(obj1, obj2):
    # 获取物体之间的距离
    r_vec = np.array(obj1.position) - np.array(obj2.position)
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)  # 避免除以零

    # 计算库仑力
    force_magnitude = COULOMB_CONSTANT * obj1.charge * obj2.charge / r ** 2

    # 方向是从 obj1 指向 obj2
    force_direction = r_vec / r

    # 返回电磁力
    return force_magnitude * force_direction
def set_object_orientation_parallel_to_ground(obj):
    """将物体的旋转角度设置为平行于地面（即绕 X 和 Y 轴的旋转角度为零）"""
    obj.rotation = (0, 0, np.random.uniform(0, 2*np.pi))  # Z 轴可以随机旋转，但 X 和 Y 保持为 0



def compute_object_height(obj):
    """根据物体的形状和 scale 计算物体的高度."""
    if obj.asset_id == "sphere":
        return obj.scale[0]  # 球体的 scale[0] 是半径，物体高度是直径
    elif obj.asset_id == "cube":
        return obj.scale[0]  # 立方体的 scale[0] 是边长，高度就是 scale[0]
    elif obj.asset_id == "cylinder":
        return obj.scale[2]  # 圆柱体的 scale[2] 是高度
    # 其他物体类型，可以继续添加逻辑处理
    else:
        return obj.scale[2]  # 默认返回 z 轴的 scale 作为高度
def set_object_on_ground(obj):
    """确保物体贴着地面生成."""
    object_height = compute_object_height(obj)
    # 将物体的位置 z 轴设置为负的 (height / 2) ，这样物体底部贴着地面
    obj.position = (obj.position[0], obj.position[1], object_height / 2)

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
# # 将对象添加到场景中
# cube1 = kubasic.create(
#     asset_id="cube",
#     name="cube",
#     scale=5.0 # 使用均匀的缩放比例,
#
# )
#
# # 设置旋转和位置
# cube1.friction = FLAGS.floor_friction
# cube1.restitution = FLAGS.floor_restitution
# cube1.material=floor_material
# floor_material.color = kb.Color.from_name("gray")
# cube1.mass = 1000.0  # 设置质量
# cube1.position = (-2, 0, -1.5)
#
# cube1.static = True  # 设置为静态物体
# scene += cube1
# cube1.background=True
# cube1_blender = cube1.linked_objects[renderer]
#
#
# cube1_blender.rotation_mode = 'XYZ'  # 确保使用正确的旋转模式
# cube1_blender.rotation_euler = (np.radians(120), 0, 0)
#
#
# # 应用变换以确保旋转在Blender中生效
# # 应用变换以确保旋转在Blender中生效
# bpy.context.view_layer.objects.active = cube1_blender
# bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


#

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

def is_overlapping(obj, existing_objects, padding=0.5):
    """检查 obj 是否与 existing_objects 中的任意物体重叠，padding 用于确保有一定的安全距离"""
    for existing_obj in existing_objects:
        # 如果 scale 是数组，取其平均值
        obj_scale = np.mean(obj.scale) if isinstance(obj.scale, (list, tuple, np.ndarray)) else obj.scale
        existing_obj_scale = np.mean(existing_obj.scale) if isinstance(existing_obj.scale, (list, tuple, np.ndarray)) else existing_obj.scale

        # 计算两物体的距离
        distance = np.linalg.norm(np.array(obj.position) - np.array(existing_obj.position))
        if distance < (obj_scale + existing_obj_scale) / 2 + padding:
            return True
    return False


existing_objects = []  # 存储已生成物体的属性
# Add random objects
num_objects = rng.randint(FLAGS.min_num_objects, FLAGS.max_num_objects + 1)
logging.info("Randomly placing %d objects:", num_objects)
generated_objects = []
for i in range(num_objects):
    properties = generate_object_properties(existing_objects, rng)

    obj = kubasic.create(
        asset_id=properties["shape_name"],
        scale=properties["size"],
        name=f"{properties['size_label']} {properties['color_label']} {properties['material_name']} {properties['shape_name']}"
    )
    obj.charge = rng.choice([-1, 1])  *9e-5   # 随机分配电荷

    assert isinstance(obj, kb.FileBasedObject)
    # 尝试生成不重叠的位置
    positioned = False
    attempts = 0
    max_attempts =  1000  # 限制尝试次数以避免无限循环

    while not positioned and attempts < max_attempts:
        # obj.position = rng.uniform(SPAWN_REGION[0], SPAWN_REGION[1])
        object_height = compute_object_height(obj)
        obj.position = (random.uniform(-5, 5), random.uniform(-5, 5), object_height / 2)
        if not is_overlapping(obj, generated_objects):  # 检查是否重叠
            # set_object_on_ground(obj)  # 调整物体位置确保贴着地面生成
            set_object_orientation_parallel_to_ground(obj)  # 确保物体平行于地面

            positioned = True
        attempts += 1

    if not positioned:
        logging.warning(f"Could not find a non-overlapping position for {obj.name} after {max_attempts} attempts.")
        continue  # 跳过当前物体，如果没有找到合适的位置

    if properties["material_name"] == "metal":
        obj.material = kb.PrincipledBSDFMaterial(color=properties["random_color"], metallic=1.0,
                                                 roughness=0.2, ior=2.5)
        obj.friction = 0.4
        # obj.friction = 0
        obj.restitution = 0.3
        obj.mass *= 2.7 * properties["size"] ** 3
    else:  # material_name == "rubber"
        obj.material = kb.PrincipledBSDFMaterial(color=properties["random_color"], metallic=0.,
                                                 ior=1.25, roughness=0.7,
                                                 specular=0.33)
        # obj.friction = 0
        obj.friction = 0.8
        obj.restitution = 0.7
        obj.mass *= 1.1 * properties["size"] ** 3

    obj.metadata = {
        "shape": properties["shape_name"].lower(),
        "size": properties["size"],
        "size_label": properties["size_label"],
        "material": properties["material_name"].lower(),
        "color": properties["random_color"].rgb,
        "color_label": properties["color_label"],
    }

    scene += obj
    kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
    # 调整生成物体的速度范围和质量

    # initialize velocity randomly but biased towards center
    obj.velocity = (rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0])
    print(f"Setting position for {obj.name}: {obj.position}, Height: {obj.scale},{object_height}")

    logging.info("    Added %s at %s", obj.asset_id, obj.position)
    existing_objects.append(properties)  # 存储已生成物体的属性
    generated_objects.append(obj)  # 添加物体到列表
    obj_blender = obj.linked_objects[renderer]

    obj_blender.rotation_mode = 'XYZ'  # 确保使用正确的旋转模式
    obj_blender.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.objects.active = obj_blender
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
#摩擦系数
dome.friction = FLAGS.floor_friction  # 设置摩擦系数
dome.restitution = FLAGS.floor_restitution  # 设置恢复系数

#物体位置
positions = []  # 用于存储每一帧的位置
orientation = []  # 用于存储每一帧的位置
output_file = 'object_positions.txt'
full_path = os.path.join(output_dir, output_file)
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
        for i, obj1 in enumerate(generated_objects):
            total_force = np.zeros(3)
            for j, obj2 in enumerate(generated_objects):
                if i != j:
                    # 计算 obj1 受到的电磁力
                    force = compute_electromagnetic_force(obj1, obj2)
                    total_force += force

            # 根据总力计算加速度并更新速度
            acceleration = total_force / obj1.mass
            new_velocity = np.array(obj1.velocity) + acceleration / FLAGS.frame_rate
            obj1.velocity = new_velocity

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
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
        mass = obj.mass
        friction = obj.friction
        restitution = obj.restitution
        file.write(f"Object {obj.name} - Mass={mass}, Friction={friction}, Restitution={restitution}\n")

# --- Rendering
if FLAGS.save_state:
    logging.info("Saving the renderer state to '%s'", output_dir / "scene.blend")
    renderer.save_state(output_dir / "scene.blend")

logging.info("Rendering the scene ...")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(visible_foreground_assets, key=lambda asset: np.sum(asset.metadata["visibility"]), reverse=True)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(data_stack["segmentation"], scene.assets, visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets)

kb.write_image_dict(data_stack, output_dir)
kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions": kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
})

kb.done()
