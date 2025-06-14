import cv2
import os
import glob

def draw_ball_trajectory_video(image_folder, label_folder, output_path, conf_threshold=0.46):
    import cv2, os, glob
    trajectory = []
    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    if not images:
        print("No images found.")
        return
    first_img = cv2.imread(images[0])
    height, width, _ = first_img.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        basename = os.path.basename(img_path).replace('.jpg', '')
        label_path = os.path.join(label_folder, basename + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, _, _, conf = map(float, line.strip().split()[:6])
                    if int(cls) == 0 and conf > conf_threshold:
                        cx = int(x * width)
                        cy = int(y * height)
                        trajectory.append((cx, cy))
        for i in range(1, len(trajectory)):
            cv2.line(img, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        out.write(img)
    out.release()
    print(f"video saved to: {output_path}")


# image_folder = '/home/louis/github/Golf_pose_eval/Golf_Ball_Trajector_Tracking/runs/detect/504-1' 
# label_folder = '/home/louis/github/Golf_pose_eval/Golf_Ball_Trajector_Tracking/runs/detect/504-1/labels' 
# output_video = '504.mp4'

# trajectory = []

# images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

# first_img = cv2.imread(images[0])
# height, width, _ = first_img.shape

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# for img_path in images:
#     img = cv2.imread(img_path)
#     basename = os.path.basename(img_path).replace('.jpg', '')
#     label_path = os.path.join(label_folder, basename + '.txt')


#     if os.path.exists(label_path):
#         with open(label_path, 'r') as f:
#             for line in f.readlines():
#                 cls, x, y, w, h , conf = map(float, line.strip().split()[:6])

#                 if int(cls) == 0 and conf > 0.46:  
#                     cx = int(x * width)
#                     cy = int(y * height)
#                     trajectory.append((cx, cy))


#     for i in range(1, len(trajectory)):
#         cv2.line(img, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

#     out.write(img)

# out.release()
# print(f"：{output_video}")
