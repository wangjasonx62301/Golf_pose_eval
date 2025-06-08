import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import *

def main():
    parser = argparse.ArgumentParser(description="Train decoder script for pose evaluation.")
    parser.add_argument("--yaml_file", type=str, default="../cfg/time_series_vae.yaml", help="Path to config yaml file")
    parser.add_argument("--skeleton_connection_file", type=str, default="../cfg/pose_connection.yaml", help="Path to skeleton connection config file")
    parser.add_argument("--mode", type=int, default=1, help="index 0,1,2")
    args = parser.parse_args()

    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.skeleton_connection_file, 'r') as f:
        skeleton_connection = yaml.safe_load(f)

    train_advice_decoder(config=config)
    
if __name__ == "__main__":
    main()