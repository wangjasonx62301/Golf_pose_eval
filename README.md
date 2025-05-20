# Golf_pose_eval

## Installation

```bash
# Clone the repository
git clone https://github.com/wangjasonx62301/Golf_pose_eval.git

# Navigate into the directory
cd Golf_pose_eval

# Install dependencies
pip install -r requirements.txt 
```

## Dataset preparation
```bash
cd script

python mp4_to_skeleton_data.py --input_video_dir your_input_folder_path --output_path your_output_folder_path --skeleton_config ../cfg/pose_connection.yaml