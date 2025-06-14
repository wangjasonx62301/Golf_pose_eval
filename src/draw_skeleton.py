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

def draw_overlay_skeleton(original_json_path, predicted_json_path, input_video_path, output_video_path, skeleton_connections):
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    with open(predicted_json_path, 'r') as f:
        predicted_data = json.load(f)

    total_frames = min(len(original_data['frames']), len(predicted_data['frames']))

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        canvas = frame.copy()
        original_kpts = original_data['frames'][frame_idx]['persons'][0]['keypoints']
        predicted_kpts = predicted_data['frames'][frame_idx]['persons'][0]['keypoints']


        orig_kpts_aligned = original_kpts


        if 11 < len(original_kpts):
            target_center = original_kpts[11][:2]  # 第11點的 (x, y)
        else:
            target_center = (width // 2, height // 2)  # fallback

        pred_kpts_aligned = align_keypoints_to_target(predicted_kpts, target_center)


        for i, j in skeleton_connections:
            if i < len(orig_kpts_aligned) and j < len(orig_kpts_aligned):
                x1, y1, _ = orig_kpts_aligned[i]
                x2, y2, _ = orig_kpts_aligned[j]
                if (x1 != 0.0 or y1 != 0.0) and (x2 != 0.0 or y2 != 0.0):
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
        for x, y, _ in orig_kpts_aligned:
            if x != 0.0 or y != 0.0:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 0), -1)

        for i, j in skeleton_connections:
            if i < len(pred_kpts_aligned) and j < len(pred_kpts_aligned):
                x1, y1, _ = pred_kpts_aligned[i]
                x2, y2, _ = pred_kpts_aligned[j]
                if (x1 != 0.0 or y1 != 0.0) and (x2 != 0.0 or y2 != 0.0):
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(canvas, pt1, pt2, (0, 0, 255), 2)
        for x, y, _ in pred_kpts_aligned:
            if x != 0.0 or y != 0.0:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)

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