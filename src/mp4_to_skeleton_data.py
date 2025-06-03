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
from src.preprocessing_json import preprocessing_json, preprocessing_json_with_linear_interpolation

def check_path_exist(path):
    
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f'Error : Path {path} not exist')
    return path

def get_single_skeleton(skeleton_connection=None, input_video_path=None, output_folder_path=None, model_name='yolo11n-pose.pt', label=None):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Cannot open video：{input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    label = label

    os.makedirs(output_folder_path, exist_ok=True)

    video_filename = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_folder_path, f"skeleton_only_{video_filename}")

    keypoints_list = {
        "video_info": {
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "label": label
        },
        "frames": []
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"start processing：{input_video_path}")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        skeleton_frame = np.zeros_like(frame)

        frame_keypoints_info = {
            "frame": frame_idx,
            "persons": []
        }


        conf_num = 0.3
        
        for r in results:
            if r.keypoints:
                keypoints = r.keypoints.xy.cpu().numpy()
                confs = r.keypoints.conf.cpu().numpy()
                
                for person_id, (person_kpts, person_confs) in enumerate(zip(keypoints, confs)):
                    dx, dy = 0, 0 

                    person_data = {
                        "id": person_id,
                        "keypoints": []
                    }

                    for i, (x, y) in enumerate(person_kpts):
                        conf = person_confs[i]
                        if conf > conf_num and not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                            person_data["keypoints"].append([float(x), float(y), float(conf)])
                        else:
                            person_data["keypoints"].append([0.0, 0.0, 0.0])

                        if conf > conf_num and x > 0 and y > 0:
                            cv2.circle(skeleton_frame, (int(x + dx), int(y + dy)), 4, (0, 0, 255), -1)

                    for i, j in skeleton_connection:
                        if (0 <= i < len(person_kpts) and 0 <= j < len(person_kpts)):
                            x1, y1 = person_kpts[i]
                            x2, y2 = person_kpts[j]
                            c1, c2 = person_confs[i], person_confs[j]

                            if (
                                c1 is not None and c1 > conf_num and not np.isnan(x1) and not np.isnan(y1) and
                                c2 is not None and c2 > conf_num and not np.isnan(x2) and not np.isnan(y2) and 
                                x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0
                            ):
                                pt1 = (int(x1 + dx), int(y1 + dy))
                                pt2 = (int(x2 + dx), int(y2 + dy))
                                cv2.line(skeleton_frame, pt1, pt2, (0, 255, 0), 2)

                    frame_keypoints_info["persons"].append(person_data)

        keypoints_list["frames"].append(frame_keypoints_info)

        out.write(skeleton_frame)
        frame_idx += 1

    cap.release()
    out.release()

    json_output_path = os.path.join(output_folder_path, f"keypoints_{os.path.splitext(video_filename)[0]}.json")
    with open(json_output_path, 'w') as f:
        json.dump(keypoints_list, f, indent=2)

    print(f"file successful output：{output_video_path}")
    print(f"keypoints saved to：{json_output_path}")
    



# def main():
#     parser = argparse.ArgumentParser(description="Output skeleton video using YOLOv11.")
#     parser.add_argument("--input_video_dir", type=str, help="Path to golf mp4.")
#     parser.add_argument("--output_path", type=str, default=None, help="Directory to save the skeleton videos.")
#     parser.add_argument('--skeleton_config', type=str, default=None, help='Skeleton Connection')

#     args = parser.parse_args()

#     with open(args.skeleton_config, 'r') as f:
#         config = yaml.safe_load(f)
#         SKELETON_CONNECTIONS = config.get('skeleton_connections')
#         if SKELETON_CONNECTIONS is None:
#             raise ValueError("Cannot find skeleton connections")



#     for filename in os.listdir(args.input_video_dir):
#         if filename.lower().endswith('.mp4'):
#             input_video_path = os.path.join(args.input_video_dir, filename)
#             get_single_skeleton(skeleton_connection=SKELETON_CONNECTIONS, input_video_path=input_video_path, output_folder_path=args.output_path)
            
#     print('Finish processing video.')
             
#     for json_file in os.listdir(args.output_path):
#         if json_file.lower().endswith('.json'):
#             input_json_path = os.path.join(args.output_path, json_file)
#             preprocessing_json_with_linear_interpolation(input_json_path=input_json_path, output_json_path=input_json_path)
            
#     for json_file in os.listdir(args.output_path):
#         if json_file.lower().endswith('.json'):
#             input_json_path = os.path.join(args.output_path, json_file)
#             preprocessing_json(input_json_path=input_json_path, output_json_path=input_json_path)
   
            
#     print('Finish processing json file.')
    
# if __name__ == "__main__":
#     main()
