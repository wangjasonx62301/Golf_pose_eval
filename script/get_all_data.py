import sys
import os
import re



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Output skeleton video using YOLOv11.")
    parser.add_argument("--yaml_file", type=str, default='../cfg/time_series_vae.yaml', help="Path to config yaml file.")
    # parser.add_argument("--json_description_file", type=str, help="Path to golf mp4.")
    # parser.add_argument("--output_path", type=str, default=None, help="Directory to save the skeleton videos.")
    parser.add_argument('--skeleton_config', type=str, default='../cfg/pose_connection.yaml', help='Skeleton Connection')
    parser.add_argument("--mode", type=int, default=1, help='index 0,1,2')


    args = parser.parse_args()

    with open(args.skeleton_config, 'r') as f:
        config = yaml.safe_load(f)
        SKELETON_CONNECTIONS = config.get('skeleton_connections')
        if SKELETON_CONNECTIONS is None:
            raise ValueError("Cannot find skeleton connections")

    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    
    data = get_mp4_path_from_json(path=config['data']['video_script_json'], mode=args.mode)
    train_data = []
    eval_data = []
    for item in data:
        video_name = os.path.basename(item["video_name"])
        base_name = os.path.splitext(video_name)[0]
        m = re.search(r"(\d+)", base_name)
        video_id = int(m.group(1)) if m else -1
        if 100 <= video_id < 300:
            eval_data.append(item)
        else:
            train_data.append(item)

    train_output = os.path.join(config['data']['skeleton_video_base_path'], f"skeleton_data_{args.mode}")
    eval_output = os.path.join(config['data']['skeleton_video_base_path'], f"skeleton_eval_{args.mode}")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(eval_output, exist_ok=True)

    get_skeleton_video_list_from_json(data=train_data, skeleton_connection=SKELETON_CONNECTIONS, output_folder_path=train_output)
    get_skeleton_video_list_from_json(data=eval_data, skeleton_connection=SKELETON_CONNECTIONS, output_folder_path=eval_output)

if __name__ == "__main__":
    main()
    # seq, label = load_json_to_dataform(path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_data/keypoints_107-1.json')
    # print(seq)
    # dataset = Keypoint_dataset(seq, 8, label=label)
    # print(len(dataset))
    # data = MultiJSONKeypointDataset(json_paths='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_data', window_size=16)
    # print(len(data))