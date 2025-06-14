import sys
import os
import subprocess
from yaml import safe_load

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


PROJECT_ROOT = "/home/louis/github/Golf_pose_eval" #path
video_id = "061"
id_mode = [f"{video_id}-0", f"{video_id}-1", f"{video_id}-2"]


OUTPUT_DIR = f"{PROJECT_ROOT}/final/{video_id}"
VIDEO_DIR = f"{PROJECT_ROOT}/dataset/video"
IMAGE_DIR = f"{OUTPUT_DIR}/video_to_image"
LABEL_DIR = f"{OUTPUT_DIR}/label_image"
WEIGHTS_PATH = f"{PROJECT_ROOT}/Golf_Ball_Trajector_Tracking/runs/train/golf5/weights/best.pt"
DETECT_SCRIPT = f"{PROJECT_ROOT}/Golf_Ball_Trajector_Tracking/detect.py"
SKELETON_JSON = f"{PROJECT_ROOT}/dataset/skeleton_data_1/keypoints_{id_mode[1]}.json"
PREDICTED_JSON = f"{PROJECT_ROOT}/final/keypoints/keypoints_{id_mode[1]}_aligned.json"
POSE_YAML = f"{PROJECT_ROOT}/cfg/pose_connection.yaml"
BALL_VIDEO = f"{OUTPUT_DIR}/{id_mode[2]}.mp4"
SKELETON_VIDEO = f"{OUTPUT_DIR}/{id_mode[1]}.mp4"
MERGED_VIDEO = f"{OUTPUT_DIR}/{id_mode[0]}.mp4"
ADVICE_JSON = f"{PROJECT_ROOT}/final/keypoints/keypoints_{id_mode[1]}_aligned.json"

from src.create import extract_frames_from_video
from src.draw_line import draw_ball_trajectory_video
from src.draw_skeleton import draw_overlay_skeleton
from src.video import merge_videos_with_advice

def run_create():
    print("run_create")
    extract_frames_from_video(
        video_dir=VIDEO_DIR,
        target_video_id=id_mode[2],
        output_root=IMAGE_DIR,
        fps=30
    )

def run_yolo_detection():
    print("run_YOLOv5")
    detect_cmd = [
        "python", DETECT_SCRIPT,
        "--weights", WEIGHTS_PATH,
        "--source", os.path.join(IMAGE_DIR, id_mode[2]),
        "--save-txt", "--save-conf", "--exist-ok",
        "--img", "640", "--conf", "0.25",
        "--name", id_mode[2],
        "--project", LABEL_DIR
    ]
    subprocess.run(detect_cmd, check=True)

def run_draw_trajectory():

    image_folder = os.path.join(LABEL_DIR, id_mode[2])
    label_folder = os.path.join(image_folder, "labels")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    draw_ball_trajectory_video(image_folder, label_folder, BALL_VIDEO, conf_threshold=0.43)
    print("draw_line")

def run_draw_skeleton():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(POSE_YAML, "r") as f:
        skeleton_cfg = safe_load(f)
    draw_overlay_skeleton(
        SKELETON_JSON,
        PREDICTED_JSON,
        f"{VIDEO_DIR}/{id_mode[1]}.mp4",
        SKELETON_VIDEO,
        skeleton_cfg["skeleton_connections"]
    )
    print("draw_skeleton")

def run_final_merge():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merge_videos_with_advice(
        skeleton_video=SKELETON_VIDEO,
        ball_video=BALL_VIDEO,
        output_video=MERGED_VIDEO,
        json_file=ADVICE_JSON,
        slow_factor=6
    )
    print("Merging videos.")

if __name__ == "__main__":
    run_create()
    run_yolo_detection()
    run_draw_trajectory()
    run_draw_skeleton()
    run_final_merge()
    print(video_id + " ok")
