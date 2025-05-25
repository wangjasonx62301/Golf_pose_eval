import torch
import json
import numpy as np
from pathlib import Path


def load_initial_sequence(json_path, window_size):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"][:window_size]
    keypoints_seq = []

    for frame in frames:
        if not frame["persons"]:
            continue
        kp = frame["persons"][0]["keypoints"]
        flat = [v for pt in kp for v in pt]
        keypoints_seq.append(flat)

    return np.array(keypoints_seq, dtype=np.float32), data["video_info"]

def convert_to_json(pred_tensor, start_frame=0):
    pred_np = pred_tensor.view(-1, 17, 3).numpy()
    result = []
    for i, frame_keypoints in enumerate(pred_np):
        frame_dict = {
            "frame": start_frame + i,
            "persons": [
                {
                    "id": 0,
                    "keypoints": frame_keypoints.tolist()
                }
            ]
        }
        result.append(frame_dict)
    return result
