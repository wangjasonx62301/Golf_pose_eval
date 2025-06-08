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
        
        inactivate_coordinates = (0, 0)
        
        left_arms = (df[i][left_arms_idx[0]], df[i][left_arms_idx[1]])
        right_arms = (df[i][right_arms_idx[0]], df[i][right_arms_idx[1]])
        left_hip = (df[i][hip_idx[0][0]], df[i][hip_idx[0][1]])
        right_hip = (df[i][hip_idx[1][0]], df[i][hip_idx[1][1]])
        
        # check positions of arms and hips
        # print(f"Frame {i}: Left Arms: {left_arms}, Right Arms: {right_arms}, Left Hip: {left_hip}, Right Hip: {right_hip}")
        if left_arms == inactivate_coordinates or right_arms == inactivate_coordinates or left_hip == inactivate_coordinates or right_hip == inactivate_coordinates:
            left_distance = 0
            right_distance = 0
        else:
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

def extract_n_frames_after_before_highest_distance(target_json_path, source_json_path, n_frames=32, config=None, mode=None):
    """
    Extract n frames before and after the frame with the highest arms-hip distance.
    """
    frame_idx = arms_hip_distance(target_json_path, config)
    # source_frame_idx = frame_idx
    source_frame_idx = arms_hip_distance(source_json_path, config)
    
    if frame_idx is None:
        print("No valid frame found.")
        return None
    
    start_frame = max(0, frame_idx - n_frames)
    source_start_frame = max(0, source_frame_idx - n_frames)
    end_frame = frame_idx + n_frames
    source_end_frame =  source_frame_idx + n_frames
    # print(f"Extracting frames from {start_frame} to {end_frame} (total {end_frame - start_frame} frames) based on frame index {frame_idx}.")
    # print(f"Source frames from {source_start_frame} to {source_end_frame} (total {source_end_frame - source_start_frame} frames) based on source frame index {soucre_frame_idx}.")
    target_df = load_json_to_dataform(target_json_path, norm=False)
    source_df = load_json_to_dataform(source_json_path, norm=False)
    
    # print(f"Extracting frames from {start_frame} to {end_frame} (total {end_frame - start_frame} frames) based on frame index {frame_idx}.")
    
    extracted_target_df = target_df[start_frame:end_frame]
    extracted_source_df = source_df[source_start_frame:source_end_frame]
    # print(f"Extracted {len(extracted_target_df)} frames from target json and {len(extracted_source_df)} frames from source json.")
    # Convert to json format
    extracted_target_df = preds_to_json_format(extracted_target_df, width=1440, height=1080, start_frame=start_frame, norm=False)
    extracted_source_df = preds_to_json_format(extracted_source_df, width=1440, height=1080, start_frame=start_frame, norm=False)
    
    if os.path.exists(f"{config['data']['extracted_json_path']}_{mode}") is False:
        os.makedirs(f"{config['data']['extracted_json_path']}_{mode}", exist_ok=True)
    if os.path.exists(f"{config['data']['extracted_json_path']}_EVAL_{mode}") is False:
        os.makedirs(f"{config['data']['extracted_json_path']}_EVAL_{mode}", exist_ok=True)
    # output_path = config['data']['extracted_json_path']
    
    save_prediction_as_json(extracted_target_df, video_info={"width": 1440, "height": 1080, "fps": 30.0}, save_path=f"{config['data']['extracted_json_path']}_{mode}/{os.path.basename(target_json_path)}")
    save_prediction_as_json(extracted_source_df, video_info={"width": 1440, "height": 1080, "fps": 30.0}, save_path=f"{config['data']['extracted_json_path']}_EVAL_{mode}/{os.path.basename(source_json_path)}")
# test
# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/cfg/pose_connection.yaml', 'r') as f:
#     skeleton_connections = yaml.safe_load(f)
#     skeleton_connections = skeleton_connections['skeleton_connections']
    
# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/cfg/time_series_vae.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# extract_n_frames_after_before_highest_distance('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1/keypoints_100-1.json', n_frames=32, config=config)
# draw_skeleton_video('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1/keypoints_100-1_extracted.json', skeleton_connections=skeleton_connections, output_path=f"{config['data']['extracted_json_path']}/{os.path.basename(json_path)}")

def extract_n_frames_from_json_folder(target_json_folder, source_json_folder, n_frames=32, config=None, mode=None):
    """
    Extract n frames before and after the frame with the highest arms-hip distance from all json files in the folder.
    """
    target_json_files = list(Path(target_json_folder).glob("*.json"))
    source_json_files = list(Path(source_json_folder).glob("*.json"))
    
    target_json_files = sorted(target_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    source_json_files = sorted(source_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    
    for target_json_file, source_json_file in zip(target_json_files, source_json_files):
        print(f"Processing {target_json_file} and {source_json_file}...")
        extract_n_frames_after_before_highest_distance(str(target_json_file), str(source_json_file), n_frames=n_frames, config=config, mode=mode)
        
    print("Extraction completed.")
    
# test 
# extract_n_frames_from_json_folder('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1', n_frames=32, config=config)


def draw_extracted_skeleton_videos(target_json_folder, source_json_folder, skeleton_connections, config, mode=None):
    """
    Draw skeleton videos from extracted json files.
    """
    if not os.path.exists(f"{config['data']['extracted_video_path']}_{mode}"):
        os.makedirs(f"{config['data']['extracted_video_path']}_{mode}", exist_ok=True)
    if not os.path.exists(f"{config['data']['extracted_video_path']}_EVAL_{mode}"):
        os.makedirs(f"{config['data']['extracted_video_path']}_EVAL_{mode}", exist_ok=True)
        
    target_json_files = list(Path(target_json_folder).glob("*.json"))
    source_json_files = list(Path(source_json_folder).glob("*.json"))
    
    target_json_files = sorted(target_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    # file name: # keypoints_100-1_extracted.json
    source_json_files = sorted(source_json_files, key=lambda x: int(x.name.split('_')[1].split('-')[0]))  # Sort by video index
    
    for target_json_file, source_json_file in zip(target_json_files, source_json_files):
        print(f"Drawing skeleton video for {target_json_file}...")
        draw_skeleton_video(str(target_json_file), skeleton_connections=skeleton_connections, output_path=f"{config['data']['extracted_video_path']}_{mode}/{os.path.basename(target_json_file).replace('.json', '.mp4')}", inference=config['data']['inference_mode'])
        
        print(f"Drawing evaluation skeleton video for {source_json_file}...")
        draw_skeleton_video(str(source_json_file), skeleton_connections=skeleton_connections, output_path=f"{config['data']['extracted_video_path']}_EVAL_{mode}/{os.path.basename(source_json_file).replace('.json', '.mp4')}", inference=1)
    
    # for json_file in json_files:
    #     print(f"Drawing skeleton video for {json_file}...")
    #     draw_skeleton_video(str(json_file), skeleton_connections=skeleton_connections, output_path=f"{config['data']['extracted_video_path']}_{mode}/{os.path.basename(json_file).replace('.json', '.mp4')}", inference=config['data']['inference_mode'])
    
    print("Drawing completed.")
    

def inference_extracted_skeleton_videos(config, skeleton_connections, mode=None):
    """
    Main function to extract frames and draw skeleton videos.
    """
    # Extract n frames before and after the frame with the highest arms-hip distance
    extract_n_frames_from_json_folder(f"{config['data']['predicted_json_path']}_{mode}", f"{config['data']['eval_json_dir']}_{mode}", n_frames=config['data']['window_size'], config=config, mode=mode)
    
    # Draw skeleton videos from extracted json files
    if not os.path.exists(f"{config['data']['extracted_json_path']}_{mode}"):
        os.makedirs(f"{config['data']['extracted_json_path']}_{mode}", exist_ok=True) 
    if not os.path.exists(f"{config['data']['extracted_json_path']}_EVAL_{mode}"): 
        os.makedirs(f"{config['data']['extracted_json_path']}_EVAL_{mode}", exist_ok=True)
    
    draw_extracted_skeleton_videos(f"{config['data']['extracted_json_path']}_{mode}", f"{config['data']['extracted_json_path']}_EVAL_{mode}", skeleton_connections=skeleton_connections, config=config, mode=mode)
        

# test 
# inference_extracted_skeleton_videos(config, mode=1)


def pose_alignment(target_json_path, source_json_path, config=None, mode=None):
    """
    Align the target pose with the source pose.
    """
    target_data = load_json_to_dataform(target_json_path, norm=False)
    source_data = load_json_to_dataform(source_json_path, norm=False)
    
    # if len(target_data) != len(source_data):
    #     print("Target and source data lengths do not match.")
    #     return
    
    aligned_frames = []
    
    x_fix_point_distance = 0
    y_fix_point_distance = 0
    # calculate the fix_point distance using right shoulder in the very first frame
    right_shoulder_idx = 6
    if len(target_data) > 0 and len(source_data) > 0 and source_data[0][right_shoulder_idx * 2] > 0:
        x_fix_point_distance = source_data[0][right_shoulder_idx * 2] - target_data[0][right_shoulder_idx * 2]
        y_fix_point_distance = source_data[0][right_shoulder_idx * 2 + 1] - target_data[0][right_shoulder_idx * 2 + 1]
    # print(f"Fix point distance: {fix_point_distance}")
    else: 
        print("Warning: Right shoulder position in the first frame of source data is not valid. Using default fix point distance of 0.")
    
    for i in range(len(target_data)):
        aligned_frame = {}
        for j in range(len(target_data[i])):
                if j % 2 == 0:
                    aligned_frame[j] = target_data[i][j] + x_fix_point_distance
                else:
                    aligned_frame[j] = target_data[i][j] + y_fix_point_distance
        aligned_frames.append(aligned_frame)
    
    aligned_frames = preds_to_json_format(aligned_frames, width=1440, height=1080, start_frame=0, norm=False)
    
    if os.path.exists(f"{config['data']['aligned_json_path']}_{mode}") is False:
        os.makedirs(f"{config['data']['aligned_json_path']}_{mode}", exist_ok=True)
    
    save_prediction_as_json(aligned_frames, video_info={"width": 1440, "height": 1080, "fps": 30.0}, save_path=f"{config['data']['aligned_json_path']}_{mode}/{os.path.basename(target_json_path).replace('.json', '_aligned.json')}")

def pose_alignment_from_json_folder(target_json_folder, source_json_folder, config=None, mode=None):
    """
    Align poses from all json files in the target folder with the source poses.
    """
    target_json_files = list(Path(target_json_folder).glob("*.json"))
    source_json_files = list(Path(source_json_folder).glob("*.json"))
    target_json_files = sorted(target_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    source_json_files = sorted(source_json_files, key=lambda x: int(x.name.split('_')[1].split('-')[0]))  # Sort by video index
    
    if len(target_json_files) != len(source_json_files):
        print("Target and source json files count do not match.")
        return
    
    for target_file, source_file in zip(target_json_files, source_json_files):
        print(f"Aligning {target_file} with {source_file}...")
        pose_alignment(str(target_file), str(source_file), config=config, mode=mode)
    
    print("Pose alignment completed.")