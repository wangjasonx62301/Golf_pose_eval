import os
import cv2

def extract_frames_from_video(video_dir, output_root, target_video_id, fps=30):
    print("Extracting frames from video...")
    os.makedirs(output_root, exist_ok=True)

    for filename in os.listdir(video_dir):
        if target_video_id in filename and filename.endswith(".mp4"):
            video_path = os.path.join(video_dir, filename)
            video_id = filename[:-4]

            output_dir = os.path.join(output_root, video_id)
            os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            interval = max(int(video_fps / fps), 1)

            frame_idx = 0
            saved_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    img_name = f"{video_id}_{saved_count:04d}.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    saved_count += 1

                frame_idx += 1

            cap.release()
            print(f"{video_id}: {saved_count} saved to {output_dir}")

    print("create complete.\n")

# video_dir = "/home/louis/github/Golf_pose_eval/Golf_Ball_Trajector_Tracking/dataset/video"     
# output_root = "/home/louis/github/Golf_pose_eval/Golf_Ball_Trajector_Tracking/dataset/videos_to_image"   
# fps = 30          

# os.makedirs(output_root, exist_ok=True)

# for filename in os.listdir(video_dir):
#     if "504-1" in filename and filename.endswith(".mp4"):
#         video_path = os.path.join(video_dir, filename)
#         video_id = filename[:-4]  

#         output_dir = os.path.join(output_root, video_id)
#         os.makedirs(output_dir, exist_ok=True)

#         cap = cv2.VideoCapture(video_path)
#         video_fps = cap.get(cv2.CAP_PROP_FPS)
#         interval = max(int(video_fps / fps), 1)

#         frame_idx = 0
#         saved_count = 0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_idx % interval == 0:
#                 img_name = f"{video_id}_{saved_count:04d}.jpg"
#                 img_path = os.path.join(output_dir, img_name)
#                 cv2.imwrite(img_path, frame)
#                 saved_count += 1

#             frame_idx += 1

#         cap.release()
#         print(f"{video_id} , {saved_count} save in {output_dir}")

# print("ok")
