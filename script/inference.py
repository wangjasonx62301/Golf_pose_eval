import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.utils import *
from src.model import *
from src.pose_extractor import *
from src.pose_criterion import *
from src.data import *
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inference script for pose evaluation.")
    parser.add_argument("--yaml_file", type=str, default="../cfg/time_series_vae.yaml", help="Path to config yaml file")
    parser.add_argument("--skeleton_connection_file", type=str, default="../cfg/pose_connection.yaml", help="Path to skeleton connection config file")
    parser.add_argument("--mode", type=int, default=1, help="index 0,1,2")
    parser.add_argument("--ckpt_path", type=str, default="/home/jasonx62301/for_python/Golf/Golf_pose_eval/ckpt/_CHose_KeypointTransformer_0.0041_epochs_200_current_200_NumLayers_8_NumEmb_128_NumHead_8_Mode_1.pt", help="Path to the model checkpoint file")
    parser.add_argument("--json_folder", type=str, default="/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_eval", help="Path to the folder containing JSON files")
    parser.add_argument("--save_path", type=str, default="/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json", help="Path to save the predicted JSON files")
    args = parser.parse_args()

    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.skeleton_connection_file, 'r') as f:
        skeleton_connection = yaml.safe_load(f)

    model = KeypointTransformer(config).to('cuda')
    model.load_state_dict(torch.load(args.ckpt_path))

    # get_predicted_mp4_from_json_folder(
    #         model=model,
    #         config=config,
    #         json_folder=f'{args.json_folder}_{args.mode}',
    #         skeleton_connections=skeleton_connection,
    #         save_path=f'{args.save_path}_{args.mode}',
    #         mode=args.mode
    #     )
    
    # inference_extracted_skeleton_videos(config, skeleton_connection['skeleton_connections'], mode=args.mode)

    # config['data']['predicted_json_path'] = f"{config['data']['eval_json_dir']}"
    # config['data']['extracted_json_path'] = f"{config['data']['extracted_json_path']}_EVAL"
    # config['data']['extracted_video_path'] = f"{config['data']['extracted_video_path']}_EVAL"
    # config['data']['inference_mode'] = 0
    # # print(f"config['data']['predicted_json_path'] = {config['data']['predicted_json_path']}")
    # # print(f"config['data']['extracted_json_path'] = {config['data']['extracted_json_path']}")
    # inference_extracted_skeleton_videos(config, skeleton_connection['skeleton_connections'], mode=args.mode)
    # pose_alignment('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/extracted_json_1/keypoints_100-1_extracted.json', '/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/extracted_json_EVAL_1/keypoints_100-1_extracted.json', config, mode=1)
    
    # pose_alignment_from_json_folder(
    #     target_json_folder=f"{config['data']['extracted_json_path']}_{args.mode}",
    #     source_json_folder=f"{config['data']['extracted_json_path']}_EVAL_{args.mode}",
    #     config=config,
    #     mode=args.mode
    # )
    
    # draw_source_and_target_together_in_one_video(
    #     source_json_path=f"{config['data']['extracted_json_path']}_{args.mode}/keypoints_100-1_extracted.json",
    #     target_json_path=f"{config['data']['extracted_json_path']}_EVAL_{args.mode}/keypoints_100-1_extracted.json",
    #     skeleton_connections=skeleton_connection['skeleton_connections'],
    #     output_path=f"{config['data']['combined_video_path']}_{args.mode}/keypoints_100-1_extracted.mp4"
    # )
    # draw_source_and_target_together_in_one_video_from_json_folder(
    #     source_json_folder=f"{config['data']['extracted_json_path']}_EVAL_{args.mode}",
    #     target_json_folder=f"{config['data']['aligned_json_path']}_{args.mode}",
    #     skeleton_connections=skeleton_connection['skeleton_connections'],
    #     output_path=f"{config['data']['combined_video_path']}_{args.mode}"
    # )
    
    calculate_keypoint_distance_with_two_json_folder(target_json_folder_path=f"{config['data']['aligned_json_path']}_{args.mode}",
                                                     source_json_folder_path=f"{config['data']['extracted_json_path']}_EVAL_{args.mode}",
                                                     config=config)
    
if __name__ == '__main__': 
    main()
#     with open('../cfg/time_series_vae.yaml', 'r') as f:
#         config = yaml.safe_load(f)
        
#     with open('../cfg/pose_connection.yaml', 'r') as f:
#         skeleton_connection = yaml.safe_load(f)

#     model = KeypointTransformer(config).to('cuda')
#     model.load_state_dict(torch.load('/home/jasonx62301/for_python/Golf/Golf_pose_eval/ckpt/Best.pt'))

# # get_predicted_mp4_from_json_folder(
# #         model=model,
# #         config=config,
# #         json_folder='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_eval_1',
# #         skeleton_connections=skeleton_connection,
# #         save_path='/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_predict_json_1',
# #     )

#     x_np, _ = load_initial_sequence('/home/jasonx62301/for_python/Golf/Golf_pose_eval/dataset/skeleton_data/keypoints_081-1.json', config['data']['window_size'])

#     x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to('cuda')  # (1, T, 51)

#     predicted = model.predict_future(x_tensor, 150, 'cuda')

# # json_output = {
# #     "frames": convert_to_json(predicted, start_frame=config['data']['window_size'])
# # }

# # with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json', 'w') as f:
# #     json.dump(json_output, f, indent=2)


# # with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json', 'r') as f:
# #     data = json.load(f)

# # if "video_info" not in data:
# #     data["video_info"] = {
# #         "width": 1440,
# #         "height": 1080,
# #         "fps": 30.0
# #     }

# # for frame in data["frames"]:
# #     # print(frame)
# #     for person in frame["persons"]:
# #         for i, keypoint in enumerate(person["keypoints"]):
# #             x, y, z = keypoint
# #             person["keypoints"][i] = [x * 1440, y * 1080, z]



# # with open('/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json', 'w') as f:
# #     json.dump(data, f, indent=2)

# video_info = {
#     "width": 1440,
#     "height": 1080,
#     "fps": 30.0
# }

# pred_frames = preds_to_json_format(predicted, width=1440, height=1080, start_frame=config['data']['window_size'])
# save_prediction_as_json(pred_frames, video_info, save_path="/home/jasonx62301/for_python/Golf/Golf_pose_eval/example_output/test.json")
