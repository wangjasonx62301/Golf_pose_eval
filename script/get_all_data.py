import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Output skeleton video using YOLOv11.")
    parser.add_argument("--yaml_file", type=str, default='../cfg/time_series_vae.yaml', help="Path to config yaml file.")
    # parser.add_argument("--json_description_file", type=str, help="Path to golf mp4.")
    # parser.add_argument("--output_path", type=str, default=None, help="Directory to save the skeleton videos.")
    parser.add_argument('--skeleton_config', type=str, default=None, help='Skeleton Connection')

    args = parser.parse_args()

    with open(args.skeleton_config, 'r') as f:
        config = yaml.safe_load(f)
        SKELETON_CONNECTIONS = config.get('skeleton_connections')
        if SKELETON_CONNECTIONS is None:
            raise ValueError("Cannot find skeleton connections")

    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    
    data = get_mp4_path_from_json(path=config['data']['video_script_json'], mode=1)
    
    get_skeleton_video_list_from_json(data=data, skeleton_connection=SKELETON_CONNECTIONS, output_folder_path=config['data']['skeleton_video_path'])
    

if __name__ == "__main__":
    main()
    # seq, label = load_json_to_dataform(path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_data/keypoints_107-1.json')
    # print(seq)
    # dataset = Keypoint_dataset(seq, 8, label=label)
    # print(len(dataset))
