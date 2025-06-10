import argparse
from ast import arg
import os
import sys
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eval import *


def main():

    parser = argparse.ArgumentParser(description="Evaluation script for pose estimation.")
    parser.add_argument("--config", type=str, default="../cfg/time_series_vae.yaml", help="Path to the config file")    
    parser.add_argument("--skeleton_connection_file", type=str, default="../cfg/pose_connection.yaml", help="Path to skeleton connection config file")
    parser.add_argument("--mode", type=int, default=1, help="Mode for evaluation (0, 1, or 2)")
    
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    with open(parser.parse_args().skeleton_connection_file, 'r') as f:
        skeleton_connection = yaml.safe_load(f)
        
    skeleton_connection = skeleton_connection['skeleton_connections']
    pred_json_path = f"{config['data']['aligned_json_path']}"
    gt_json_path = f"{config['data']['eval_json_dir']}"
    
    
    metrics = evaluate_json_folder(pred_json_path, gt_json_path, skeleton_connection=skeleton_connection)
    
    best_metrics = find_best_metrics(metrics)
    avg_metrics = get_mean_metrics(metrics)
    
    # print(f"Best Metrics: {best_metrics}")
    # print(f"Average Metrics: {avg_metrics}")
    
if __name__ == "__main__":
    main()
    