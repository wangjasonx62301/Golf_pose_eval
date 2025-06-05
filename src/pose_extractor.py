import json
from turtle import left
import numpy as np
import os
from src.data import *
from src.utils import *
import yaml

def calculate_distance(point1 : tuple, point2 : tuple):
    return (abs(point1[0] - point2[0])**2 + abs(point1[1] - point2[1])**2)**0.5

def arms_hip_distance(json_path, config=None):
    
    left_arms_idx, right_arms_idx, hip_idx = (18, 19), (20, 21), ((22, 23), (24, 25)) # left_hip, right_hip
    
    df = load_json_to_dataform(json_path, norm=False)
    
    max_len = 0
    
    frame_idx = None
    
    for i in range(len(df)):
        
        left_arms = (df[i][left_arms_idx[0]], df[i][left_arms_idx[1]])
        right_arms = (df[i][right_arms_idx[0]], df[i][right_arms_idx[1]])
        left_hip = (df[i][hip_idx[0][0]], df[i][hip_idx[0][1]])
        right_hip = (df[i][hip_idx[1][0]], df[i][hip_idx[1][1]])
        
        # check positions of arms and hips
        # print(f"Frame {i}: Left Arms: {left_arms}, Right Arms: {right_arms}, Left Hip: {left_hip}, Right Hip: {right_hip}")
        
        left_distance = calculate_distance(left_arms, left_hip)
        right_distance = calculate_distance(right_arms, right_hip)
        
        total_distance = left_distance + right_distance
        # print(f"Frame {i}: Left Arms-Hip Distance: {left_distance}, Right Arms-Hip Distance: {right_distance}, Total: {total_distance}")
        if total_distance > max_len:
            max_len = total_distance
            frame_idx = i
    
    
    return frame_idx
    
# frame_dix = arms_hip_distance('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1/keypoints_100-1.json')
# print(f"Frame index with max arms-hip distance: {frame_dix}")

def extract_n_frames_after_before_highest_distance(json_path, n_frames=32, config=None, mode=None):
    """
    Extract n frames before and after the frame with the highest arms-hip distance.
    """
    frame_idx = arms_hip_distance(json_path, config)
    
    if frame_idx is None:
        print("No valid frame found.")
        return None
    
    start_frame = max(0, frame_idx - n_frames)
    end_frame = frame_idx + n_frames
    
    df = load_json_to_dataform(json_path, norm=False)
    
    # print(f"Extracting frames from {start_frame} to {end_frame} (total {end_frame - start_frame} frames) based on frame index {frame_idx}.")
    
    extracted_frames = df[start_frame:end_frame]
    
    extracted_frames = preds_to_json_format(extracted_frames, width=1440, height=1080, start_frame=0, norm=False)
    
    if os.path.exists(f"{config['data']['extracted_json_path']}_{mode}") is False:
        os.makedirs(f"{config['data']['extracted_json_path']}_{mode}", exist_ok=True)
    
    # output_path = config['data']['extracted_json_path']
    
    save_prediction_as_json(extracted_frames, video_info={"width": 1440, "height": 1080, "fps": 30.0}, save_path=f"{config['data']['extracted_json_path']}_{mode}/{os.path.basename(json_path).replace('.json', '_extracted.json')}")
    
# test
# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/cfg/pose_connection.yaml', 'r') as f:
#     skeleton_connections = yaml.safe_load(f)
#     skeleton_connections = skeleton_connections['skeleton_connections']
    
# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/cfg/time_series_vae.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# extract_n_frames_after_before_highest_distance('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1/keypoints_100-1.json', n_frames=32, config=config)
# draw_skeleton_video('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1/keypoints_100-1_extracted.json', skeleton_connections=skeleton_connections, output_path=f"{config['data']['extracted_json_path']}/{os.path.basename(json_path)}")

def extract_n_frames_from_json_folder(json_folder, n_frames=32, config=None, mode=None):
    """
    Extract n frames before and after the frame with the highest arms-hip distance from all json files in the folder.
    """
    json_files = list(Path(json_folder).glob("*.json"))
    
    for json_file in json_files:
        print(f"Processing {json_file}...")
        extract_n_frames_after_before_highest_distance(str(json_file), n_frames=n_frames, config=config, mode=mode)
    
    print("Extraction completed.")
    
# test 
# extract_n_frames_from_json_folder('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1', n_frames=32, config=config)


def draw_extracted_skeleton_videos(json_folder, skeleton_connections, config, mode=None):
    """
    Draw skeleton videos from extracted json files.
    """
    if not os.path.exists(f"{config['data']['extracted_video_path']}_{mode}"):
        os.makedirs(f"{config['data']['extracted_video_path']}_{mode}", exist_ok=True)
    
    json_files = list(Path(json_folder).glob("*.json"))
    
    for json_file in json_files:
        print(f"Drawing skeleton video for {json_file}...")
        draw_skeleton_video(str(json_file), skeleton_connections=skeleton_connections, output_path=f"{config['data']['extracted_video_path']}_{mode}/{os.path.basename(json_file).replace('.json', '.mp4')}")
    
    print("Drawing completed.")
    

def inference_extracted_skeleton_videos(config, skeleton_connections, mode=None):
    """
    Main function to extract frames and draw skeleton videos.
    """
    # Extract n frames before and after the frame with the highest arms-hip distance
    extract_n_frames_from_json_folder(f"{config['data']['predicted_json_path']}_{mode}", n_frames=config['data']['window_size'], config=config, mode=mode)
    
    # Draw skeleton videos from extracted json files
    if not os.path.exists(f"{config['data']['extracted_json_path']}_{mode}"):
        os.makedirs(f"{config['data']['extracted_json_path']}_{mode}", exist_ok=True) 
    
    draw_extracted_skeleton_videos(f"{config['data']['extracted_json_path']}_{mode}", skeleton_connections=skeleton_connections, config=config, mode=mode)
        

# test 
# inference_extracted_skeleton_videos(config, mode=1)
