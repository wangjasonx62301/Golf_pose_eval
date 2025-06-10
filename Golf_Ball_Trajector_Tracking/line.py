import cv2
import os
import glob

# 設定資料夾
image_folder = '/home/louis/github/yolov5/runs/detect/detect_ball'  # 放圖片的資料夾
label_folder = '/home/louis/github/yolov5/runs/detect/detect_ball/labels'  # 替換成你的實際 label 資料夾
output_video = 'output_video.mp4'

# 初始化儲存球的中心點軌跡
trajectory = []

# 取得所有圖片
images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

# 載入第一張來取得尺寸
first_img = cv2.imread(images[0])
height, width, _ = first_img.shape

# 建立影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# 開始逐張處理
for img_path in images:
    img = cv2.imread(img_path)
    basename = os.path.basename(img_path).replace('.jpg', '')
    label_path = os.path.join(label_folder, basename + '.txt')

    # 如果該張有標記
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split()[:5])

                if int(cls) == 0:  # 只處理 ball 類別
                    cx = int(x * width)
                    cy = int(y * height)
                    trajectory.append((cx, cy))

    # 畫出所有歷史球點軌跡
    for i in range(1, len(trajectory)):
        cv2.line(img, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

    out.write(img)

out.release()
print(f"✅ 輸出完成：{output_video}")
