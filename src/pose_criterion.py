# from data import *
from json import load
from re import S
from turtle import distance
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
    


def combine_aligned_json_with_keypoint_distance(config, aligned_json_path, keypoint_distance_json_path, model=None):
    """
    Combine aligned JSON files with keypoint distance JSON files by inserting one shared 'advice' per frame into each person's data.
    """

    combined_data = {
        "video_info": {
            "width": 1440,
            "height": 1080,
            "fps": 30.0
        },
        "frames": []
    }
    
    keypoint_keydict = {
        "5": "Category: Left Shoulder;",
        "6": "Category: Right Shoulder;",
        "7": "Category: Left Arm;",
        "8": "Category: Right Arm;"
    }
    advice_keydict = {
        1: " CorrectionDirection: lower, Correction:",
        2: " CorrectionDirection: higher, Correction:",
        0: ""
    }
    
    
    # this function only combine two json files, one is aligned json, the other is keypoint distance json
    # load json files
    with open(aligned_json_path, 'r') as f:
        aligned_data = json.load(f)

    with open(keypoint_distance_json_path, 'r') as f:
        distance_data = json.load(f)

    # print(aligned_data)
    for frame_index, frame in enumerate(aligned_data['frames']):
        # print(frame_index)
        if frame_index >= len(distance_data):
            print(f"Warning: Frame index {frame_index} exceeds distance data length. Skipping this frame.")
            continue
        advice_value = {}
        for key, value in distance_data[frame_index].items():
            if not value:
                continue
            advice_value[key] = model.generate_advice(max_new_tokens=64, input_seq=keypoint_keydict[str(key)] + advice_keydict[value])

        # print(frame)
        for person in frame["persons"]:
            person["advice"] = advice_value

        combined_data["frames"].append(frame)

    output_path = config['data']['keypoint_combined_json_path']
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{os.path.basename(aligned_json_path).split('.')[0]}_combined.json")

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"Combined JSON saved to: {output_file}")
    
# with open('../cfg/time_series_vae.yaml', 'r') as f:
#     config = yaml.safe_load(f)    

# combine_aligned_json_with_keypoint_distance(config=config, 
#                                             aligned_json_path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/aligned_json_1/keypoints_100-1_aligned.json',
#                                             keypoint_distance_json_path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/keypoint_distance_json/keypoints_100-1_aligned.json')
    
def combine_aligned_json_with_keypoint_distance_folder(config, aligned_json_folder_path, keypoint_distance_json_folder_path, model=None):
    """
    Combine aligned JSON files with keypoint distance JSON files in folders.
    """
    aligned_json_files = list(Path(aligned_json_folder_path).glob("*.json"))
    keypoint_distance_json_files = list(Path(keypoint_distance_json_folder_path).glob("*.json"))
    
    # sort the files to ensure they match
    aligned_json_files = sorted(aligned_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    keypoint_distance_json_files = sorted(keypoint_distance_json_files, key=lambda x: int(x.name.split('_')[1].split('-')[0]))  # Sort by video index
    
    if len(aligned_json_files) != len(keypoint_distance_json_files):
        raise ValueError("The number of JSON files in both folders must be the same.")
    
    for aligned_file, distance_file in zip(aligned_json_files, keypoint_distance_json_files):
        combine_aligned_json_with_keypoint_distance(config=config,
                                                    aligned_json_path=aligned_file,
                                                    keypoint_distance_json_path=distance_file,
                                                    model=model)
    print(f"Combined JSON files saved to: {config['data']['keypoint_combined_json_path']}")
  
# test  
# with open('../cfg/time_series_vae.yaml', 'r') as f:
#     config = yaml.safe_load(f)    
# combine_aligned_json_with_keypoint_distance_folder(config=config,
#                                                     aligned_json_folder_path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/aligned_json_1',
#                                                     keypoint_distance_json_folder_path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/keypoint_distance_json')

# def add_advice_in_json(config, model, json_file_path):
#     model.eval()
#     frames = {}
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
    
#     keypoints_dict = {
#         "5" : "Left Shoulder",
#         "6" : "Right Shoulder",
#         "7" : "Left Arm",
#         "8" : "Right Arm"
#     }
    
#     for frame in data:
        
        