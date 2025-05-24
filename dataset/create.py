import json

index = 901
total = 4
dataset = []
for i in range(index, index + total):
    filename = f"dataset/video/{i:03}.mp4"
    label = True if i <= 132 else False
    dataset.append({
        "video_name": filename,
        "label": label
    })

with open("dataset_1_1.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("k")
