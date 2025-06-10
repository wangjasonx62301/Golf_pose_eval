import os
import cv2

video_dir = "/home/louis/github/yolov5/dataset/video1/"       # 放影片的資料夾
output_root = "/home/louis/github/yolov5/dataset/videos_to_image_test"     # 擷取圖片輸出資料夾
fps = 30                   # 每秒擷取幾張圖片

os.makedirs(output_root, exist_ok=True)

for filename in os.listdir(video_dir):
    if filename.endswith("-2.mp4") :
        video_path = os.path.join(video_dir, filename)
        video_id = filename[:-4]  # 去掉 .mp4，例如 '001-2'

        # 建立同名資料夾，例如 images/001-2/
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
        print(f"[✓] {video_id} → {saved_count} 張儲存於 {output_dir}")

print("🎉 所有 -2.mp4 已完成擷取")
