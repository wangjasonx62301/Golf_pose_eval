import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.train import *

model = train_vae(cfg_path='../cfg/time_series_vae.yaml')