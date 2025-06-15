import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Golf")

        self.folder_path = ""
        self.image_list = []
        self.current_img = None

        # 下拉選單 + 按鈕
        self.select_btn = ttk.Button(
            root, text="Select folder", command=self.select_folder
        )
        self.select_btn.pack(pady=10)

        self.combo = ttk.Combobox(root, state="readonly")
        self.combo.bind("<<ComboboxSelected>>", self.folder_selected)
        self.combo.pack()

        # 圖片顯示區
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        # 拉桿
        self.slider = tk.Scale(
            root, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_image
        )
        self.slider.pack(fill="x")

    def select_folder(self):
        parent_dir = filedialog.askdirectory(title="Select root folder")
        if parent_dir:
            subfolders = [
                f
                for f in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, f))
            ]
            self.combo["values"] = subfolders
            self.combo.set("Select video")
            self.parent_dir = parent_dir

    def folder_selected(self, event):
        folder_name = self.combo.get()
        self.folder_path = os.path.join(self.parent_dir, folder_name)
        self.image_list = sorted(
            [
                f
                for f in os.listdir(self.folder_path)
                if f.endswith((".png", ".jpg", ".jpeg")) and f.split(".")[0].isdigit()
            ],
            key=lambda x: int(x.split(".")[0]),
        )

        if self.image_list:
            self.slider.config(from_=0, to=len(self.image_list) - 1)
            self.slider.set(0)
            self.show_image(0)

    def update_image(self, val):
        index = int(val)
        self.show_image(index)

    def show_image(self, index):
        image_path = os.path.join(self.folder_path, self.image_list[index])
        img = Image.open(image_path)
        img = img.resize((800, 600), Image.Resampling.LANCZOS)
        self.current_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_img)


# 執行
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
