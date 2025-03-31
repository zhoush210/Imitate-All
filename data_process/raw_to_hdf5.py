import convert_all as crd
import os
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task_name", type=str)
parser.add_argument("-vn", "--video_names", type=str, nargs="+")
parser.add_argument("-ds", "--downsampling", type=int, default=0)
parser.add_argument("-md", "--mode", type=str, default="real3")
parser.add_argument("-pad", "--padding", action="store_true")
parser.add_argument("-dir", "--raw_dir", type=str, default="data/raw")
parser.add_argument("-segment", "--segment", action="store_true", help="If set, use videos from output directory instead of raw directory")
# parser.add_argument("-bson", "--use_bson_style", action="store_true")
args = parser.parse_args()

task_name = args.task_name
downsampling = args.downsampling
mode = args.mode
padding = args.padding
raw_dir = args.raw_dir
segment = args.segment
# use_bson_style = args.use_bson_style

task_dir = os.path.abspath(f"{raw_dir}/{task_name}")
assert os.path.exists(task_dir), f"task_dir {task_dir} not exists"
# raw_dir = os.path.abspath(raw_dir)
# assert os.path.exists(raw_dir)

# 如果设置了segment参数，视频数据将从output目录读取
if segment:
    video_dir = os.path.abspath(f"data/output/{task_name}")
    assert os.path.exists(video_dir), f"video_dir {video_dir} not exists"
else:
    video_dir = task_dir


def load_raw_real_data(raw_dir, downsampling=0):
    data = crd.raw_to_dict(
        raw_dir,
        ["low_dim.json"],
        video_file_names=None,
        flatten_mode=None,
        concatenater={
            "/observations/qpos": (
                "observation/arm/joint_position",
                "observation/eef/joint_position",
            ),
            "/action": (
                "action/arm/joint_position",
                "action/eef/joint_position",
                # "observation/base/velocity",
            ),
        },
        key_filter=[
            "observation/eef/pose",
            "action/eef/pose",
            # "/time",
        ],
    )
    return crd.downsample(data, downsampling)


def load_raw_mujoco_data(raw_dir, downsampling=0):
    data = crd.raw_to_dict(
        raw_dir,
        ["obs_action.json"],
        video_file_names=None,
        flatten_mode="hdf5",
        name_converter={
            "/obs/jq": "/observations/qpos",
            "/act": "/action",
        },
        pre_process=None,
        concatenater=None,
        key_filter=["/time"],
    )
    return crd.downsample(data, downsampling)


# load raw low dim data
if mode == "real3":
    low_dim_data = load_raw_real_data(task_dir, downsampling)
elif mode == "mujoco":
    low_dim_data = load_raw_mujoco_data(task_dir, downsampling)
else:
    raise ValueError(f"mode {mode} is not supported")

# merge high_dim data and save
raw_names = args.video_names
video_names = [name + ".mp4" for name in raw_names]
target_dir = f"data/hdf5/{task_name}/"
name_converter = {
    raw_names[i]: f"/observations/images/{i}" for i in range(len(raw_names))
}
print(f"name_converter: {name_converter}")
target_namer = lambda i: f"episode_{i}.hdf5"
compresser = crd.Compresser("jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 50], True)

# get max episode length
episode_lens = []
for low_d in low_dim_data.values():
    episode_lens.append(len(list(low_d.values())[0]))
max_pad_length = max(episode_lens) if padding else None

print(f"Episode flatten keys: {list(low_dim_data.values())[0].keys()}")
episode_names = list(low_dim_data.keys())
print(f"Episode number: {len(episode_names)}")
print(f"Max episode length: {max_pad_length}")
print(f"All episodes: {episode_names}")

# create target dir
os.makedirs(target_dir, exist_ok=True)


def save_one(index, ep_name):
    # 动作数据依然从原目录读取
    low_dim = low_dim_data[ep_name]
    
    # 视频数据根据segment参数决定从哪个目录读取
    video_path = f"{video_dir}/{ep_name}" if segment else f"{task_dir}/{ep_name}"
    
    # 获取动作数据长度
    action_length = len(list(low_dim.values())[0])
    
    # 检查所有视频的帧数
    video_lengths = []
    for video_name in video_names:
        video_file = os.path.join(video_path, video_name)
        if os.path.exists(video_file):
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                video_lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                cap.release()
            else:
                print(f"Warning: Could not open video {video_file}")
                video_lengths.append(float('inf'))  # 使用无穷大表示无法打开的视频
        else:
            print(f"Warning: Video file not found: {video_file}")
            video_lengths.append(float('inf'))  # 使用无穷大表示不存在的视频
    
    # 计算有效长度（所有视频和动作数据的最小长度）
    min_video_length = min(video_lengths) if video_lengths else 0
    valid_length = min(action_length, min_video_length)
    
    if valid_length <= 0:
        print(f"Error: Episode {ep_name} has no valid frames. Skipping.")
        low_dim_data.pop(ep_name)
        return
    
    # 如果有长度不匹配，输出警告信息
    if action_length != min_video_length:
        print(f"Warning: Length mismatch in {ep_name}:")
        print(f"  - Action data length: {action_length}")
        print(f"  - Video lengths: {video_lengths}")
        print(f"  - Using valid length: {valid_length}")
    
    # 截断动作数据到有效长度
    truncated_low_dim = {}
    for key, value in low_dim.items():
        truncated_low_dim[key] = value[:valid_length]
    
    # 创建临时视频文件夹，存储截断后的视频
    if action_length != min_video_length:
        temp_dir = os.path.join(os.path.dirname(video_path), f"temp_{ep_name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 重新写入截断后的视频
        temp_video_names = []
        for video_name in video_names:
            src_video = os.path.join(video_path, video_name)
            dst_video = os.path.join(temp_dir, video_name)
            
            if os.path.exists(src_video):
                cap = cv2.VideoCapture(src_video)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(dst_video, fourcc, fps, (width, height))
                    
                    frame_count = 0
                    while cap.isOpened() and frame_count < valid_length:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        frame_count += 1
                    
                    cap.release()
                    out.release()
                    temp_video_names.append(video_name)
                    print(f"Created truncated video: {dst_video} with {frame_count} frames")
            
        # 使用临时视频目录进行处理
        if temp_video_names:
            video_path = temp_dir
            print(f"Using temporary truncated videos from {temp_dir}")
    
    # 使用截断后的数据进行处理
    crd.merge_video_and_save(
        truncated_low_dim,
        video_path,
        video_names,
        crd.save_dict_to_hdf5,
        name_converter,
        compresser,
        f"{target_dir}/" + target_namer(index),
        max_pad_length,
        downsampling,
    )
    
    # 清理临时文件
    if action_length != min_video_length and 'temp_dir' in locals() and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    
    low_dim_data.pop(ep_name)


# save all data
print(f"Start saving all data to {target_dir}...")
futures = []
with ThreadPoolExecutor(max_workers=25) as executor:
    for index, ep_name in enumerate(episode_names):
        futures.append(executor.submit(save_one, index, ep_name))
print(f"All data saved to {target_dir}")

# # save one data
# index = 0
# ep_name = episode_names[index]
# save_one(index, ep_name)