import torch
import json
import numpy as np
from pathlib import Path
import cv2
import os

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



def draw_skeleton_video(json_path, skeleton_connections, output_path):
    with open(json_path, 'r') as f:
        keypoints_data = json.load(f)

    video_info = keypoints_data["video_info"]
    frames_data = keypoints_data["frames"]

    frame_size = (video_info["width"], video_info["height"])
    fps = video_info["fps"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame_info in frames_data:
        canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        for person in frame_info["persons"]:
            keypoints = person["keypoints"]

            # Draw keypoints
            for x, y, conf in keypoints:
                
                cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)

            # Draw skeleton lines
            for i, j in skeleton_connections:
                if i < len(keypoints) and j < len(keypoints):
                    x1, y1, c1 = keypoints[i]
                    x2, y2, c2 = keypoints[j]

                    # if c1 > 0.3 and c2 > 0.3:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

        out.write(canvas)

    out.release()
    print(f"skeleton video saved to: {output_path}")
