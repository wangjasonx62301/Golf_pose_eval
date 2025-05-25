import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.train import *

model = train_classifier(vae_ckpt="/home/jasonx62301/for_python/Golf/Golf_pose_eval/ckpt/Time_Series_VAE_1.8613958448841004_epochs_50.pt", cfg_path='../cfg/time_series_vae.yaml')