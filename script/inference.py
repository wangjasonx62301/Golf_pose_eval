import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.utils import *
from src.model import *
import yaml

with open('../cfg/time_series_vae.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = KeypointTransformer(config).to('cuda')
model.load_state_dict(torch.load('/home/jasonx62301/for_python/Golf/Golf_pose_eval/ckpt/KeypointTransformer_1.9242_epochs_200.pt'))

x_np, _ = load_initial_sequence('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_data/keypoints_015-1.json', config['data']['window_size'])

x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to('cuda')  # (1, T, 51)

predicted = model.predict_future(x_tensor, 150, 'cuda')

# json_output = {
#     "frames": convert_to_json(predicted, start_frame=config['data']['window_size'])
# }

# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json', 'w') as f:
#     json.dump(json_output, f, indent=2)


# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json', 'r') as f:
#     data = json.load(f)

# if "video_info" not in data:
#     data["video_info"] = {
#         "width": 1440,
#         "height": 1080,
#         "fps": 30.0
#     }

# for frame in data["frames"]:
#     # print(frame)
#     for person in frame["persons"]:
#         for i, keypoint in enumerate(person["keypoints"]):
#             x, y, z = keypoint
#             person["keypoints"][i] = [x * 1440, y * 1080, z]



# with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json', 'w') as f:
#     json.dump(data, f, indent=2)

video_info = {
    "width": 1440,
    "height": 1080,
    "fps": 30.0
}

pred_frames = preds_to_json_format(predicted, width=1440, height=1080, start_frame=config['data']['window_size'])
save_prediction_as_json(pred_frames, video_info, save_path="/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json")
