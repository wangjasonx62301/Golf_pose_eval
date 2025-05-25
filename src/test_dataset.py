import os
import json
import argparse
from mp4_to_skeleton_data import get_single_skeleton
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_json', type=str, required=True, help='Path to video_dataset.json')
    parser.add_argument('--skeleton_config', type=str, required=True, help='Path to pose_connection.yaml')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save keypoints_XXX.json')
    parser.add_argument('--model_path', type=str, default='yolo11n-pose.pt', help='Path to YOLO pose model')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.label_json, 'r') as f:
        label_data = json.load(f)

    with open(args.skeleton_config, 'r') as f:
        sk_config = yaml.safe_load(f)
        skeleton_connection = sk_config.get("skeleton_connections")
        if skeleton_connection is None:
            raise ValueError("skeleton_connections not found in config")

    for item in label_data:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        video_path = os.path.join(project_root, item["video_name"])
        print(f"now：{video_path}")

        if not os.path.exists(video_path):
            print(f"NOT VIDEO：{video_path}")
            continue

        get_single_skeleton(
            skeleton_connection=skeleton_connection,
            input_video_path=video_path,
            output_folder_path=args.output_dir,
            model_name=args.model_path
        )


    print("K")

if __name__ == '__main__':
    main()
