import json
from logging import config
import numpy as np
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import sys
import tiktoken

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from src.pose_criterion import *
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
        


def load_json_to_dataform(path=None, norm=True):

    path = check_path_exist(path)
    
    with open(path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    # label = data['video_info']['label']
    keypoints_seq = []
    for frame in frames:
        person = frame["persons"][0]  
        keypoints = person["keypoints"]
        flat = []
        for pt in keypoints:
            # brute force normalize
            # may need fix
            if pt[0] > 0 and pt[1] > 0 and not np.isnan(pt[0]) and not np.isnan(pt[1]):
                pt[0] = pt[0]
                pt[1] = pt[1]
                
                if norm == True:
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
    def __init__(self, json_paths, min_input_len=32, max_input_len=200):
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

def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def encode_text(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer.encode(text, allowed_special="all")

def decode_text(token_ids, tokenizer=None):
    """
    Decode a sequence of token IDs back to text.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    return tokenizer.decode(token_ids)

def process_correction_dataset(csv_path):
    df = pd.read_csv(csv_path)

    dataset = []
    for _, row in df.iterrows():
        category = row["Category"].strip()
        direction = row["CorrectionDirection"].strip()
        correction = row["Correction"].strip()
        if direction == "neutral/adjust": continue
        input_text = f"Category: {category}; CorrectionDirection: {direction}"
        target_text = correction

        dataset.append({
            "input": input_text,
            "target": target_text
        })
    return dataset

def df_to_text_sequence(df, tokenizer=None):
    """
    Convert a DataFrame to a text sequence.
    Each row is converted to a string and then tokenized.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # extract text from "Correction" column if it existsdf = df.copy()
    
    text_sequence = []
    # add end token in every period
    for text in df:
        # preserve index
        # This is text format
        # {
        #     "input": "Category: Right Arm; CorrectionDirection: lower",
        #     "target": "Keep your right arm tucked in."
        # }
        # concat input and target text
        concat_text = f"{text['input']}, Correction: {text['target']}"
        if concat_text:
            tokens = encode_text(concat_text, tokenizer)
            # Add end token to each text sequence
            tokens.append(tokenizer.eot_token)
            # Add begin token in very first token
            tokens.insert(0, 100259)
            text_sequence.append(tokens)
        else:
            text_sequence.append([])
    
    return text_sequence

    
def get_df(csv_path):
    
    csv_path = check_path_exist(csv_path)
    return pd.read_csv(csv_path, encoding='utf-8', dtype=str).fillna('')

def get_text_token_sequence_from_csv(csv_path, tokenizer=None):
    """
    Read a CSV file and convert the 'Correction' column to a text token sequence.
    """
    df = process_correction_dataset(csv_path)

    return df_to_text_sequence(df, tokenizer)

class AdviceDataset(Dataset):
    def __init__(self, config, tokenizer, df):
        
        self.config = config
        self.tokenizer = tokenizer
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        target = self.df[idx]
        target = torch.tensor(target, dtype=torch.long)
        
        # padding
        if len(target) < self.config['data']['max_seq_len']:
            target = F.pad(target, (0, self.config['data']['max_seq_len'] - len(target)), value=self.config['data']['pad_token_id'])
        else:
            target = target[:self.config['data']['max_seq_len']]
            
        return target
    
def get_advice_batch(df, target_idx, config):
    
    if target_idx == None:
        target_idx = np.random.randint(0, len(df) - 1)

    ix = torch.randint(len(df[target_idx]) - config['advice_model']['block_size'], (config['advice_model']['batch_size'],))
        
    x = torch.stack([df[target_idx][i:i + config['advice_model']['block_size']] for i in ix], dim=0)
    y = torch.stack([df[target_idx][i + 1:i + config['advice_model']['block_size'] + 1] for i in ix], dim=0)
    x, y = x.to(config['data']['device']), y.to(config['data']['device'])
    return x, y

# class Pose_Advice_Keypoint_Label(Dataset):
    
#     def __init__(self, data):
#         super().__init__()
#         if data is None:
#             self.data = get_df()