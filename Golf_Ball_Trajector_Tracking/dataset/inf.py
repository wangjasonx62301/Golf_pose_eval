import cv2
import os
import natsort

image_folder = '/home/louis/github/Golf_pose_eval/Golf_Ball_Trajector_Tracking/runs/detect/504-1'
label_folder = os.path.join(image_folder, 'labels')
output_video = '504.mp4'
fps = 30
conf_threshold = 0.46
target_class_id = 0


images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images = natsort.natsorted(images)

if not images:
    raise ValueError("no image")


first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

trajectory = []

for img_name in images:
    image_path = os.path.join(image_folder, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_folder, label_name)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f" {image_path}，跳過")
        continue

    ball_center = None
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                class_id = int(parts[0])
                conf = float(parts[1])
                if class_id == target_class_id and conf > conf_threshold:
                    x_center = float(parts[2]) * width
                    y_center = float(parts[3]) * height
                    ball_center = (int(x_center), int(y_center))
                    break 

    if ball_center:
        trajectory.append(ball_center)


    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

    video_writer.write(frame)

video_writer.release()
print(f"：{output_video}")
