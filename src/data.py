import json
import numpy as np
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.mp4_to_skeleton_data import get_single_skeleton

def check_path_exist(path):
    
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f'Error : Path {path} not exist')
    return path

def get_mp4_path_from_json(path=None, mode=0):
    
    path += f'{mode}.json'
    path = check_path_exist(path)

    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
        
    formatted_data = [
        {"video_name": item["video_name"], "label": bool(item["label"])}
        for item in data
    ]
    
    return formatted_data
    
def get_skeleton_video_list_from_json(data=None, skeleton_connection=None, output_folder_path=None):
    
    label_dict = {
        True: 1,
        False: 0
    }
    
    output_folder_path = check_path_exist(output_folder_path)
    
    for row in data:
        video_name = row["video_name"]
        label = label_dict[row["label"]]
        
        input_video_path = os.path.join("../", video_name)
        if not os.path.exists(input_video_path):
            print(f"Video {input_video_path} does not exist, skipping.")
            continue
        
        print(f"Processing video: {video_name}")
        get_single_skeleton(skeleton_connection, input_video_path, output_folder_path, label=label)
        


def load_json_to_dataform(path=None):

    path = check_path_exist(path)
    
    with open(path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    label = data['video_info']['label']
    keypoints_seq = []
    for frame in frames:
        person = frame["persons"][0]  
        keypoints = person["keypoints"]
        flat = []
        for pt in keypoints:
            # brute force normalize
            # may need fix
            pt[0] /= data['video_info']['width']
            pt[1] /= data['video_info']['height']
            flat.extend(pt)  # x, y, c
            # print(flat)
        keypoints_seq.append(flat)

    keypoints_seq = np.array(keypoints_seq)  
    return keypoints_seq, label

class Keypoint_dataset(Dataset):
    def __init__(self, seq, window_size, label=None):
        super().__init__()
        self.window_size = window_size
        self.sequences = []
        self.targets = [] 
        for i in range(len(seq) - window_size): 
            self.sequences.append(seq[i:i+window_size])
            self.targets.append(seq[i+window_size]) 
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(1) 
        self.label = label

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.label
    

    
