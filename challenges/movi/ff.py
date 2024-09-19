import os
import subprocess

# 设置图像帧所在目录和输出视频文件的路径
frames_dir = r"D:\download\kubric-main\output"
output_video = r"D:\download\kubric-main\output\video.mp4"

# 确保frames_dir路径存在
if not os.path.exists(frames_dir):
    raise FileNotFoundError(f"Frames directory '{frames_dir}' not found")

# 查找帧图像文件，假设它们是按顺序命名的，比如 frame0001.png, frame0002.png, ...
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
if not frame_files:
    raise FileNotFoundError("No frame images found in the specified directory")

# 构建ffmpeg命令
ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "30",  # 设置帧率，可以根据需要调整
    "-i", os.path.join(frames_dir, "rgba_%05d.png"),  # 输入文件模式
    "-c:v", "libx264",  # 使用H.264编码
    "-pix_fmt", "yuv420p",  # 设置像素格式
    "-crf", "18",  # 设置恒定质量因子，数值越小质量越高
    output_video
]

# 执行ffmpeg命令
subprocess.run(ffmpeg_cmd, check=True)


