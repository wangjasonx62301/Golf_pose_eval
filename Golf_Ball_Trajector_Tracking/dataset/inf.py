import cv2
import os
from pathlib import Path
import natsort  # 確保圖片排序自然，例如 1.jpg, 2.jpg, ..., 10.jpg

# === 設定路徑 ===
image_folder = '/home/louis/github/yolov5/runs/detect/detect_ball'  # 放圖片的資料夾
output_video = 'output_video.mp4'      # 輸出的影片檔案
fps = 30                               # 每秒幾張圖（影片播放速度）

# === 讀取圖片檔名並排序 ===
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images = natsort.natsorted(images)  # 自然排序：1.jpg, 2.jpg, ..., 10.jpg

if not images:
    raise ValueError("找不到任何圖片！請確認資料夾與副檔名")

# === 取得圖片尺寸 ===
first_img_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_img_path)
height, width, _ = frame.shape

# === 初始化影片寫入器 ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === 將每張圖片寫入影片 ===
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"無法讀取圖片 {image_path}，跳過")
        continue
    video_writer.write(frame)

video_writer.release()
print(f"✅ 合成完成！影片輸出至：{output_video}")
