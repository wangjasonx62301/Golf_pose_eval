# from data import *
from requests import get
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.utils import *
from src.data import *
import tiktoken
import pandas as pd

# encoding = tiktoken.get_encoding("gpt2


# # test
# batch_x = get_text_token_sequence_from_csv("/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/Pose_advice/pose_advice.csv", tokenizer=get_tokenizer())
# print(decode_text(batch_x[0], get_tokenizer()))  # Print first 10 tokens for verification
# print(batch_x[:9])  # Print first 10 token IDs for verification
# print(max(len(x) for x in batch_x))  # Print the maximum length of token sequences

def get_df_category(df):
    if 'Category' not in df.columns:
        raise ValueError("'Category' column not found in DataFrame")
    # Extract unique categories
    categories = df['Category'].unique()
    # Create a dictionary mapping each category to its index
    category_to_index = {category: index for index, category in enumerate(categories)}
    # Create a new column 'CategoryIndex' in the DataFrame
    df['CategoryIndex'] = df['Category'].map(category_to_index)
    # Return the modified DataFrame and the category mapping
    return df, category_to_index

def calculate_distance(keypoint_1, keypoint_2):
    """
    Calculate the Euclidean distance between two keypoints.
    Each keypoint is a tuple (x, y).
    """
    return keypoint_2 - keypoint_1

def calculate_keypoint_distance_with_two_json(config, target_json_file_path, source_json_file_path):
    """
    Calculate the distance between two sets of keypoints from JSON files.
    """
    keypoints_1 = load_json_to_dataform(target_json_file_path, norm=False)
    keypoints_2 = load_json_to_dataform(source_json_file_path, norm=False)
    # print(keypoints_1.shape, keypoints_2.shape)
    new_frames = []
    length = min(len(keypoints_1), len(keypoints_2))
    for i in range(length):
        
        frame = {}
        
        valid_keypoints = [5, 6, 7, 8]
        
        for j in range(len(keypoints_1[i])):
            
            if keypoints_1[i][j] is None or keypoints_2[i][j] is None:
                continue
            
            if  j % 2 and (j - 1) // 2 in valid_keypoints:
                
                # focus on y_coordinate, used to get the current keypoint is lower or higher than false source
                # Calculate the distance between the two keypoints
                dis = calculate_distance(keypoints_1[i][j], keypoints_2[i][j])
                # store new data
                if abs(dis) > config['data']['keypoint_distance_threshold']:
                    
                    if dis < 0:
                        frame[(j - 1) // 2] = 1 # source is lower than target
                    else:  
                        frame[(j - 1) // 2] = 2 # source is higher than target
                        
                else: frame[(j - 1) // 2] = 0 # source is similar to target
                
        if frame:
            new_frames.append(frame)
    # Save the new frames to a JSON file
    if not os.path.exists(f"{config['data']['keypoint_distance_json_path']}"):
        os.makedirs(f"{config['data']['keypoint_distance_json_path']}", exist_ok=True)
    with open(f"{config['data']['keypoint_distance_json_path']}/{os.path.basename(target_json_file_path)}", 'w') as f:
        json.dump(new_frames, f, indent=2)
    # print(f"Keypoint distance JSON saved to: {config['data']['keypoint_distance_json_path']}")
    # return new_frames
    
    

def calculate_keypoint_distance_with_two_json_folder(config, target_json_folder_path, source_json_folder_path):
    """
    Calculate the distance between two sets of keypoints from JSON files in folders.
    """
    target_json_files = list(Path(target_json_folder_path).glob("*.json"))
    source_json_files = list(Path(source_json_folder_path).glob("*.json"))
    
    # sort the files to ensure they match
    target_json_files = sorted(target_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    source_json_files = sorted(source_json_files, key=lambda x: int(x.name.split('_')[1].split('-')[0]))  # Sort by video index
    # print(f"Target JSON files: {target_json_files}")
    # print(f"Source JSON files: {source_json_files}")
    if len(target_json_files) != len(source_json_files):
        raise ValueError("The number of JSON files in both folders must be the same.")
    
    # all_frames = []
    
    for target_file, source_file in zip(target_json_files, source_json_files):
        # target_file_path = os.path.join(target_json_folder_path, target_file)
        # source_file_path = os.path.join(source_json_folder_path, source_file)
        
        calculate_keypoint_distance_with_two_json(config, target_file, source_file)
        # save json file
    print(f"Keypoint distance JSON files saved to: {config['data']['keypoint_distance_json_path']}")
    


