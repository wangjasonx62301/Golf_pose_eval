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
            if pt[0] > 0 and pt[1] > 0 and not np.isnan(pt[0]) and not np.isnan(pt[1]):
                pt[0] /= data['video_info']['width']
                pt[1] /= data['video_info']['height']

            else:
                pt[0] = 0.0
                pt[1] = 0.0

            flat.extend((pt[0], pt[1]))  # x, y, c
            # print(flat)
        keypoints_seq.append(flat)

    keypoints_seq = np.array(keypoints_seq)  
    return keypoints_seq

class Keypoint_dataset(Dataset):
    def __init__(self, seq, window_size):
        super().__init__()
        self.window_size = window_size
        self.sequences = []
        self.targets = []

        for i in range(len(seq) - window_size):
            self.sequences.append(seq[i:i + window_size])
            self.targets.append(seq[i + window_size])  

        self.sequences = torch.from_numpy(np.array(self.sequences)).float()
        self.targets = torch.from_numpy(np.array(self.targets)).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
    
class MultiJSONKeypointDataset(Dataset):
    def __init__(self, json_paths, window_size):
        self.sequences = []
        self.targets = []
        all_jsons = [os.path.join(json_paths, f) for f in os.listdir(json_paths) if f.endswith(".json")]
        for path in all_jsons:
            keypoints_seq = load_json_to_dataform(path)

            for i in range(len(keypoints_seq) - window_size):
                self.sequences.append(keypoints_seq[i:i+window_size])
                self.targets.append(keypoints_seq[i+window_size])

        self.sequences = torch.from_numpy(np.array(self.sequences)).float()
        self.targets = torch.from_numpy(np.array(self.targets)).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
    
class AutoRegressiveKeypointDataset(Dataset):
    def __init__(self, json_paths, min_input_len=32, max_input_len=180):
        self.samples = []

        all_jsons = [os.path.join(json_paths, f) for f in os.listdir(json_paths) if f.endswith(".json")]

        for path in all_jsons:
            keypoints_seq = load_json_to_dataform(path)  # shape: (T, 34)

            T = len(keypoints_seq)
            for i in range(min_input_len, min(T, max_input_len)):
                input_seq = keypoints_seq[:i]       # shape: (i, 34)
                target = keypoints_seq[i]           # shape: (34,)
                self.samples.append((input_seq, target))

        # find max input length to pad to
        self.max_len = max_input_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        seq_len = input_seq.shape[0]
        
        # Zero-padding to max length
        padded = np.zeros((self.max_len, input_seq.shape[1]), dtype=np.float32)
        padded[:seq_len] = input_seq

        # attention mask for transformer (1 = real token, 0 = padding)
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        return (
            torch.tensor(padded),           # input_seq: (max_len, 34)
            torch.tensor(mask),            # attention_mask: (max_len,)
            torch.tensor(target).float()   # target: (34,)
        )


