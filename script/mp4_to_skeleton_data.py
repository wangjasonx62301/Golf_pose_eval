from fileinput import filename
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
import yaml


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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"start processing：{input_video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        skeleton_frame = np.zeros_like(frame)

        for r in results:
            if r.keypoints:
                keypoints = r.keypoints.xy.cpu().numpy()  # shape: (n_persons, 17, 2)
                confs = r.keypoints.conf.cpu().numpy()    # shape: (n_persons, 17)

                for person_kpts, person_confs in zip(keypoints, confs):
                    dx, dy = 0, 0 

                    for i, (x, y) in enumerate(person_kpts):
                        if person_confs[i] > 0.3:
                            cv2.circle(skeleton_frame, (int(x + dx), int(y + dy)), 4, (0, 0, 255), -1)

                    for i, j in skeleton_connection:
                        if (0 <= i < len(person_kpts) and 0 <= j < len(person_kpts)):
                            x1, y1 = person_kpts[i]
                            x2, y2 = person_kpts[j]
                            c1, c2 = person_confs[i], person_confs[j]

                            if (
                                c1 is not None and c1 > 0.3 and not np.isnan(x1) and not np.isnan(y1) and (x1, y1) != (0, 0) and
                                c2 is not None and c2 > 0.3 and not np.isnan(x2) and not np.isnan(y2) and (x2, y2) != (0, 0)
                            ):
                                pt1 = (int(x1 + dx), int(y1 + dy))
                                pt2 = (int(x2 + dx), int(y2 + dy))
                                cv2.line(skeleton_frame, pt1, pt2, (0, 255, 0), 2)
        out.write(skeleton_frame)

    cap.release()
    out.release()
    print(f"file successful output：{output_video_path}")
    
     

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
            
    print('Finish processing.')
    
if __name__ == "__main__":
    main()
