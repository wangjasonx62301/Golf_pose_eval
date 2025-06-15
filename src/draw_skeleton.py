import cv2
import json
import numpy as np
import yaml
import os

def align_keypoints_to_target(keypoints, target_point, center_idx=11):
    if center_idx >= len(keypoints):
        return keypoints
    cx, cy, _ = keypoints[center_idx]
    dx = target_point[0] - cx
    dy = target_point[1] - cy
    return [[x + dx, y + dy, c] for x, y, c in keypoints]

def draw_kpts(canvas, kpts, color, connections):
    for i, j in connections:
        if i < len(kpts) and j < len(kpts):
            x1, y1, _ = kpts[i]
            x2, y2, _ = kpts[j]
            if (x1 != 0.0 or y1 != 0.0) and (x2 != 0.0 or y2 != 0.0):
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.line(canvas, pt1, pt2, color, 2)
    for x, y, _ in kpts:
        if x != 0.0 or y != 0.0:
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1)

def shift_predicted_json(predicted_data, offset, original_path):
    shifted = dict(predicted_data)

    empty_person = [{"id": 0, "keypoints": [[0.0, 0.0, 0.0]] * 17}]
    empty_frames = [{"frame": i, "persons": empty_person} for i in range(offset)]

    new_frames = []
    for i, frame in enumerate(predicted_data["frames"]):
        new_frame = dict(frame)
        new_frame["frame"] = i + offset
        new_frames.append(new_frame)

    shifted["frames"] = empty_frames + new_frames

    base, ext = os.path.splitext(original_path)
    shifted_path = f"{base}_shifted{ext}"

    with open(shifted_path, 'w') as f:
        json.dump(shifted, f, indent=2)

    print(f"Shifted predicted JSON saved to: {shifted_path}")



def draw_overlay_skeleton(
    original_json_path,
    predicted_json_path,
    input_video_path,
    output_video_path,
    skeleton_connections,
    frame_offset=0
):
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    with open(predicted_json_path, 'r') as f:
        predicted_data = json.load(f)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predicted_frames = predicted_data['frames']
    if frame_offset > 0:
        empty_person = [{"id": 0, "keypoints": [[0.0, 0.0, 0.0]] * 17}]
        empty_frames = [{"frame": i, "persons": empty_person} for i in range(frame_offset)]
        predicted_frames = empty_frames + predicted_frames
    elif frame_offset < 0:
        predicted_frames = predicted_frames[abs(frame_offset):]

    if len(predicted_frames) < total_frames:
        pad_len = total_frames - len(predicted_frames)
        empty_person = [{"id": 0, "keypoints": [[0.0, 0.0, 0.0]] * 17}]
        predicted_frames += [{"frame": i, "persons": empty_person} for i in range(pad_len)]
    else:
        predicted_frames = predicted_frames[:total_frames]

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if frame_offset > 0:
        shift_predicted_json(predicted_data, frame_offset, predicted_json_path)


    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = frame.copy()
        original_kpts = []
        predicted_kpts = []

        if frame_idx < len(original_data['frames']):
            original_kpts = original_data['frames'][frame_idx]['persons'][0]['keypoints']

        predicted_kpts = predicted_frames[frame_idx]['persons'][0]['keypoints']


        if original_kpts:
            draw_kpts(canvas, original_kpts, (0, 255, 0), skeleton_connections)
            if 11 < len(original_kpts):
                target_center = original_kpts[11][:2]
            else:
                target_center = (width // 2, height // 2)
        else:
            target_center = (width // 2, height // 2)

        if predicted_kpts:
            pred_kpts_aligned = align_keypoints_to_target(predicted_kpts, target_center)
            draw_kpts(canvas, pred_kpts_aligned, (0, 0, 255), skeleton_connections)

        out.write(canvas)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"skeleton video saved to: {output_video_path}")

# def main():
#     original_json_path = "/home/louis/github/Golf_pose_eval/final/keypoints_063-1.json"
#     predicted_json_path = "/home/louis/github/Golf_pose_eval/final/keypoints_063-1_aligned.json"
#     input_video_path = "/home/louis/github/Golf_pose_eval/final/keypoints_063-1.mp4"
#     output_video_path = "/home/louis/github/Golf_pose_eval/final/overlay_result1_063-1.mp4"
#     skeleton_yaml_path = "../cfg/pose_connection.yaml"

#     with open(skeleton_yaml_path, 'r') as f:
#         sk_cfg = yaml.safe_load(f)
#         skeleton_connections = sk_cfg.get("skeleton_connections")
#         if skeleton_connections is None:
#             raise ValueError("skeleton_connections not found in YAML")

#     draw_overlay_skeleton(
#         original_json_path,
#         predicted_json_path,
#         input_video_path,
#         output_video_path,
#         skeleton_connections
#     )

# if __name__ == "__main__":
#     main()