import json
import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def check_path_exist(path):
    
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f'Error : Path {path} not exist')
    return path

def load_json_to_dataform(path=None):

    path = check_path_exist(path)
    
    with open(path, "r") as f:
        data = json.load(f)

    frames = data["frames"]

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
    return keypoints_seq

class Keypoint_dataset(Dataset):
    def __init__(self, seq, window_size):
        super().__init__()
        self.window_size = window_size
        self.sequences = []
        for i in range(len(seq) - window_size + 1):
            self.sequences.append(seq[i:i+window_size])
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
# seq = load_json_to_dataform(path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/keypoints_081-1.json')
# dataset = Keypoint_dataset(seq, 8)
# print(len(dataset))
    
