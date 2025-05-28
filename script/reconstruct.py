import os
import yaml
import json
import cv2
import numpy as np
import copy

def load_skeleton_connections(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['skeleton_connections']

def draw_skeleton(frame, keypoints, connections, confidence_thresh=0.1):
    drawn_count = 0
    for i, (x, y, conf) in enumerate(keypoints):
        if (
            isinstance(x, (int, float)) and isinstance(y, (int, float)) and
            not np.isnan(x) and not np.isnan(y) and
            abs(x) < 1e6 and abs(y) < 1e6 and
            conf > confidence_thresh
        ):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            drawn_count += 1

    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1, c1 = keypoints[i]
            x2, y2, c2 = keypoints[j]
            if all([
                isinstance(v, (int, float)) and not np.isnan(v) and abs(v) < 1e6
                for v in [x1, y1, x2, y2]
            ]) and c1 > confidence_thresh and c2 > confidence_thresh:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

def json_to_video(data, yaml_path, output_path, width=1440, height=1080, fps=30):
    keypoints_frames = data["frames"]
    connections = load_skeleton_connections(yaml_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_data in keypoints_frames:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for person in frame_data["persons"]:
            draw_skeleton(frame, person["keypoints"], connections, confidence_thresh=-1.0)
        out.write(frame)

    out.release()
    print(f"video output path: {output_path}")

with open('../cfg/time_series_vae.yaml', 'r') as f:
    config = yaml.safe_load(f)

json_path = os.path.abspath('../example_output/test.json')
yaml_path = os.path.abspath('../cfg/pose_connection.yaml')
output_path = os.path.abspath(config['output']['reconstructed_video_path'])
scaled_json_path = json_path.replace(".json", "_scaled.json")

default_width = 1440
default_height = 1080
default_fps = 30

with open(json_path, 'r') as f:
    data = json.load(f)

if "video_info" not in data:
    data["video_info"] = {
        "width": default_width,
        "height": default_height,
        "fps": default_fps
    }

width = int(data["video_info"]["width"])
height = int(data["video_info"]["height"])
fps = int(data["video_info"]["fps"])

scaled_data = copy.deepcopy(data)


for frame in scaled_data["frames"]:
    for person in frame["persons"]:
        for i, keypoint in enumerate(person["keypoints"]):
            x, y, conf = keypoint
            if (
                isinstance(x, (int, float)) and isinstance(y, (int, float)) and
                not np.isnan(x) and not np.isnan(y) and
                abs(x) < 10 and abs(y) < 10
            ):
                person["keypoints"][i] = [x * width, y * height, conf]


with open(scaled_json_path, 'w') as f:
    json.dump(scaled_data, f, indent=2)

json_to_video(scaled_data, yaml_path, output_path, width=width, height=height, fps=fps)
