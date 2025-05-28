import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.utils import *
from src.model import Time_Series_VAE, Golf_Pose_Classifier
import yaml

with open('../cfg/time_series_vae.yaml', 'r') as f:
    config = yaml.safe_load(f)

vae = Time_Series_VAE(config).to('cuda')
vae.load_state_dict(torch.load('/home/louis/github/Golf_pose_eval/ckpt/Time_Series_VAE_1.4988378290290711_epochs_100.pt'))

x_np, _ = load_initial_sequence('/home/louis/github/Golf_pose_eval/dataset/skeleton_data_1/keypoints_034-1.json', config['data']['window_size'])

x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to('cuda')  # (1, T, 51)

predicted = vae.predict_future_frames(x_tensor, 150)

json_output = {
    "frames": convert_to_json(predicted, start_frame=config['data']['window_size'])
}

with open('/home/louis/github/Golf_pose_eval/example_output/test.json', 'w') as f:
    json.dump(json_output, f, indent=2)


with open('/home/louis/github/Golf_pose_eval/example_output/test.json', 'r') as f:
    data = json.load(f)

if "video_info" not in data:
    data["video_info"] = {
        "width": 1440,
        "height": 1080,
        "fps": 30.0
    }

for frame in data["frames"]:
    # print(frame)
    for person in frame["persons"]:
        for i, keypoint in enumerate(person["keypoints"]):
            x, y, z = keypoint
            person["keypoints"][i] = [x * 1440, y * 1080, z]



with open('/home/louis/github/Golf_pose_eval/example_output/test.json', 'w') as f:
    json.dump(data, f, indent=2)
