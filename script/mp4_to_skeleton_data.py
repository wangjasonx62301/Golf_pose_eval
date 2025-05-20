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

def check_path_exist(path):
    
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f'Error : Path {path} not exist')
    return path

def get_single_skeleton(skeleton_connection=None, input_video_path=None, output_folder_path=None, model_name='yolo11n-pose.pt'):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Cannot open video：{input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(output_folder_path, exist_ok=True)

    video_filename = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_folder_path, f"skeleton_only_{video_filename}")

    # 🔧 keypoints info will be stored here
    keypoints_list = {
        "video_info": {
            "width": frame_width,
            "height": frame_height,
            "fps": fps
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
                        person_data["keypoints"].append([float(x), float(y), float(conf)])
                        if conf > 0.3:
                            cv2.circle(skeleton_frame, (int(x + dx), int(y + dy)), 4, (0, 0, 255), -1)

                    for i, j in skeleton_connection:
                        if (0 <= i < len(person_kpts) and 0 <= j < len(person_kpts)):
                            x1, y1 = person_kpts[i]
                            x2, y2 = person_kpts[j]
                            c1, c2 = person_confs[i], person_confs[j]

                            if (
                                c1 is not None and c1 > 0.3 and not np.isnan(x1) and not np.isnan(y1) and
                                c2 is not None and c2 > 0.3 and not np.isnan(x2) and not np.isnan(y2)
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

def main():
    parser = argparse.ArgumentParser(description="Output skeleton video using YOLOv11.")
    parser.add_argument("--input_video_dir", type=str, help="Path to golf mp4.")
    parser.add_argument("--output_path", type=str, default=None, help="Directory to save the skeleton videos.")
    parser.add_argument('--skeleton_config', type=str, default=None, help='Skeleton Connection')

    args = parser.parse_args()

    with open(args.skeleton_config, 'r') as f:
        config = yaml.safe_load(f)
        SKELETON_CONNECTIONS = config.get('skeleton_connections')
        if SKELETON_CONNECTIONS is None:
            raise ValueError("Cannot find skeleton connections")



    for filename in os.listdir(args.input_video_dir):
        if filename.lower().endswith('.mp4'):
            input_video_path = os.path.join(args.input_video_dir, filename)
            get_single_skeleton(skeleton_connection=SKELETON_CONNECTIONS, input_video_path=input_video_path, output_folder_path=args.output_path)
            
    print('Finish processing video.')
             
    # for json_file in os.listdir(args.output_path):
    #     if json_file.lower().endswith('.json'):
    #         input_json_path = os.path.join(args.output_path, json_file)
    #         preprocessing_json_with_linear_interpolation(input_json_path=input_json_path, output_json_path=input_json_path)
            
    # for json_file in os.listdir(args.output_path):
    #     if json_file.lower().endswith('.json'):
    #         input_json_path = os.path.join(args.output_path, json_file)
    #         preprocessing_json(input_json_path=input_json_path, output_json_path=input_json_path)
   
            
    # print('Finish processing json file.')
    
if __name__ == "__main__":
    main()
