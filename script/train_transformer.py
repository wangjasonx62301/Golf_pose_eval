from ast import arg
import sys
import os
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.train import train_transformer, train_transformer_AR



def main():
    parser = argparse.ArgumentParser(description="Train Transformer model for pose evaluation.")
    parser.add_argument("--yaml_file", type=str, default="../cfg/time_series_vae.yaml", help="Path to config yaml file")
    parser.add_argument("--mode", type=int, default=1, help="index 0,1,2")
    args = parser.parse_args()

    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['json_dir'] = os.path.join(config['data']['skeleton_video_base_path'], f"skeleton_data")
    config['data']['eval_json_dir'] = os.path.join(config['data']['skeleton_video_base_path'], f"skeleton_eval_{args.mode}")

    model = train_transformer(cfg_path=None, config=config, mode=args.mode)
    
if __name__ == '__main__':
    main()
    