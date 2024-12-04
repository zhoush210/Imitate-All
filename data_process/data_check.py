import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task_name", type=str, default="example")
parser.add_argument(
    "-vn", "--video_names", type=str, nargs="+", default=["observation.images.cam1.mp4"]
)
parser.add_argument("-ep", "--episode", type=int, default=0)
args = parser.parse_args()


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        duration = None
    else:
        duration = frame_count / fps
    return frame_count, duration


for name in args.video_names:
    video_path = f"data/raw/{args.task_name}/{args.episode}/{name}"
    frame_count, duration = get_video_length(video_path)
    print(f"{name}: {frame_count} frames {duration} seconds")
