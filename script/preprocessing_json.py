from fileinput import filename
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
import yaml
import json
from sklearn.impute import KNNImputer
import pandas as pd

def preprocessing_json(input_json_path, output_json_path, k=3):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    num_frames = len(frames)
    num_keypoints = 17  
    max_persons = max(len(f['persons']) for f in frames)

    for person_id in range(max_persons):
        for kp_idx in range(num_keypoints):
            xs = []
            ys = []
            valids = []

            for frame in frames:
                person = next((p for p in frame['persons'] if p['id'] == person_id), None)
                if person:
                    x, y, c = person['keypoints'][kp_idx]
                    if (x == 0.0 and y == 0.0) or c < 0.3:
                        xs.append(np.nan)
                        ys.append(np.nan)
                        valids.append(False)
                    else:
                        xs.append(x)
                        ys.append(y)
                        valids.append(True)
                else:
                    xs.append(np.nan)
                    ys.append(np.nan)
                    valids.append(False)

            arr = np.array([xs, ys]).T  # shape: (frames, 2)
            
            if np.isnan(arr[:, 0]).all() or np.isnan(arr[:, 1]).all():
                continue
            imputer = KNNImputer(n_neighbors=k, weights="distance")
            imputed = imputer.fit_transform(arr)

            for f_idx, frame in enumerate(frames):
                person = next((p for p in frame['persons'] if p['id'] == person_id), None)
                if person:
                    old_x, old_y, conf = person['keypoints'][kp_idx]
                    if not valids[f_idx]:  
                        new_x, new_y = imputed[f_idx]
                        person['keypoints'][kp_idx] = [float(new_x), float(new_y), 0.01]  

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Save new json to：{output_json_path}")


def preprocessing_json_with_linear_interpolation(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    frames = data["frames"]
    num_kpts = len(frames[0]["persons"][0]["keypoints"])
    person_ids = sorted({p["id"] for f in frames for p in f["persons"]})

    # Interpolate for each person and each keypoint index
    for pid in person_ids:
        for kpt_idx in range(num_kpts):
            xs, ys, confs = [], [], []

            for frame in frames:
                person = next((p for p in frame["persons"] if p["id"] == pid), None)
                if person:
                    x, y, c = person["keypoints"][kpt_idx]
                    xs.append(np.nan if x == 0.0 else x)
                    ys.append(np.nan if y == 0.0 else y)
                    confs.append(c)
                else:
                    xs.append(np.nan)
                    ys.append(np.nan)
                    confs.append(0.0)

            df = pd.DataFrame({'x': xs, 'y': ys})
            df_interp = df.interpolate(method='linear', limit_direction='both')

            # Replace missing values
            for i, frame in enumerate(frames):
                person = next((p for p in frame["persons"] if p["id"] == pid), None)
                if person:
                    x, y, c = person["keypoints"][kpt_idx]
                    if x == 0.0 or y == 0.0:
                        new_x = float(df_interp.iloc[i]["x"])
                        new_y = float(df_interp.iloc[i]["y"])
                        person["keypoints"][kpt_idx] = [new_x, new_y, 0.01]  # low confidence for imputed

    # Save interpolated result
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Save new json to：{output_json_path}")