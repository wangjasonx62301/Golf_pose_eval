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
        
        keypoints_seq.append(flat[:34])

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
        # add frame idx in top-right
        # cv2.putText(canvas, f"Frame: {frame_info['frame']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for person in frame_info["persons"]:
            keypoints = person["keypoints"]

            # Draw keypoints
            for x, y, conf in keypoints:
                # add position information above the keypoint
                # if conf > 0.3:  # Confidence threshold
                cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)
                # cv2.putText(canvas, f"({int(x)}, {int(y)})", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
    

def preds_to_json_format(preds, width, height, start_frame=0, default_confidence=0.0, norm=True):
    preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    frames = []

    for i, flat in enumerate(preds):
        keypoints = []
        for j in range(17):
            if norm:
                x = float(flat[2 * j] * width)      # ⬅️ 確保是 Python float
                y = float(flat[2 * j + 1] * height)
                c = float(default_confidence)
            else:
                x = float(flat[2 * j])
                y = float(flat[2 * j + 1])
                c = float(default_confidence)
            keypoints.append([x, y, c])

        frame_data = {
            "frame": int(start_frame + i),
            "persons": [
                {
                    "id": 0,
                    "keypoints": keypoints
                }
            ]
        }
        frames.append(frame_data)
    
    return frames


def save_prediction_as_json(pred_frames, video_info, save_path="output.json"):
    json_data = {
        "video_info": video_info,  # e.g. {"width": 1440, "height": 1080, "fps": 30.0}
        "frames": pred_frames
    }
    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=2)
        
def get_predicted_mp4_from_json_folder(model, config, json_folder, skeleton_connections, save_path, mode):
    
    if not os.path.exists(config['data']['predicted_video_path']):
        os.makedirs(config['data']['predicted_video_path'], exist_ok=True)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    json_files = list(Path(json_folder).glob("*.json"))
    
    video_info = config.get('video_info')
    
    for json_file in json_files:
        
        x_np, _ = load_initial_sequence(str(json_file), config['data']['window_size'])

        x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to('cuda')  # (1, T, 51)

        predicted = model.predict_future(x_tensor, 250, 'cuda')
        
        pred_frames = preds_to_json_format(predicted, width=1440, height=1080, start_frame=config['data']['window_size'])
        save_prediction_as_json(pred_frames, video_info, save_path=f'{save_path}/{json_file.stem}.json')
    
    json_files = list(Path(save_path).glob("*.json"))
    
    for json_file in json_files:
        draw_skeleton_video(str(json_file), skeleton_connections['skeleton_connections'], f"{config['data']['predicted_video_path']}_{mode}/{str(json_file)[-20:].replace('.json', '.mp4')}")
        
        
        
    print(f"All videos saved to {config['data']['predicted_video_path']}")