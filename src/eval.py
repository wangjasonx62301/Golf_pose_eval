from torch import gt
from src.utils import *
from src.data import *
import numpy as np
import math

def calculate_limb_l2_distance(keypoint1, keypoint2):
    
    return math.sqrt((keypoint1[0] - keypoint2[0]) ** 2 + (keypoint1[1] - keypoint2[1]) ** 2)

def percentage_of_correct_parts(pred_json_path, gt_json_path, skeleton_connection=None):
    """
    Calculate the percentage of correctly predicted parts.
    
    Parameters:
    pred_json_path (str): Path to the predicted JSON file.
    gt_json_path (str): Path to the ground truth JSON file.
    
    Returns:
    float: Percentage of correct parts.
    """
    pred_json = load_json_to_dataform(pred_json_path, norm=False)
    gt_json = load_json_to_dataform(gt_json_path, norm=False)
    
    total_PCP = 0
    total_len = 0
    for i in range(min(len(pred_json), len(gt_json))):
        pred_frame = pred_json[i]
        gt_frame = gt_json[i]
        cur_PCP = 0
        for connection in skeleton_connection:
            pred_keypoint1 = (pred_frame[connection[0] * 2], pred_frame[connection[0] * 2 + 1])
            pred_keypoint2 = (pred_frame[connection[1] * 2], pred_frame[connection[1] * 2 + 1])
            gt_keypoint1 = (gt_frame[connection[0] * 2], gt_frame[connection[0] * 2 + 1])
            gt_keypoint2 = (gt_frame[connection[1] * 2], gt_frame[connection[1] * 2 + 1])
            
            # gt_limb_dis
            l = calculate_limb_l2_distance(gt_keypoint1, gt_keypoint2)
            if l == 0:
                continue
            d1 = calculate_limb_l2_distance(pred_keypoint1, gt_keypoint1) / l
            d2 = calculate_limb_l2_distance(pred_keypoint2, gt_keypoint2) / l
            total_len += 1
            threshold = l * 0.5
            if ((d1 + d2) / 2) <= threshold:
                cur_PCP += 1
                
        total_PCP += cur_PCP 
    delta = total_PCP / total_len
    print(f"Total PCP: {delta * 100:.2f}%")
    return delta

def percent_of_detected_joints(pred_json_path, gt_json_path, alphas=[0.05, 0.1, 0.2, 0.5]):
    """
    Calculate the percentage of detected joints.
    
    Parameters:
    pred_json_path (str): Path to the predicted JSON file.
    gt_json_path (str): Path to the ground truth JSON file.
    
    """
    pred_json = load_json_to_dataform(pred_json_path, norm=False)
    gt_json = load_json_to_dataform(gt_json_path, norm=False)
    torso_connection = (5, 11)
    
    PDJ_005 = []
    PDJ_010 = []
    PDJ_015 = []
    PDJ_020 = []
    PDJ = [PDJ_005, PDJ_010, PDJ_015, PDJ_020]
    
    for i in range(min(len(pred_json), len(gt_json))):
        pred_frame = pred_json[i]
        gt_frame = gt_json[i]
        torso_point1 = (gt_frame[torso_connection[0] * 2], gt_frame[torso_connection[0] * 2 + 1])
        torso_point2 = (gt_frame[torso_connection[1] * 2], gt_frame[torso_connection[1] * 2 + 1])
        current_gt_torso_length = calculate_limb_l2_distance(torso_point1, torso_point2)
        for k in range(len(alphas)):
            for j in range(0, len(pred_frame), 2):
                pred_keypoint = (pred_frame[j], pred_frame[j + 1])
                gt_keypoint = (gt_frame[j], gt_frame[j + 1])
                if gt_keypoint == (0, 0):
                    # If the ground truth keypoint is (0, 0), we consider it as not detected
                    # PDJ[k].append(0)
                    continue
                d_j = calculate_limb_l2_distance(pred_keypoint, gt_keypoint)
                if d_j <= current_gt_torso_length * alphas[k]:
                    PDJ[k].append(1)
                else:
                    PDJ[k].append(0)
            
        # for pred_keypoint, gt_keypoint in zip(pred_frame, gt_frame):
        #     for j, alpha in enumerate(alphas):
        #         d_j = calculate_limb_l2_distance(pred_keypoint, gt_keypoint)
        #         if d_j <= current_gt_torso_length * alpha:
        #             PDJ[j].append(1)
        #         else:
        #             PDJ[j].append(0)
    # print(PDJ)
    PDJ = [sum(p) / len(p) for p in PDJ]
    print(f"PDJ@0.05: {PDJ[0] * 100:.2f}%")
    print(f"PDJ@0.10: {PDJ[1] * 100:.2f}%")
    print(f"PDJ@0.20: {PDJ[2] * 100:.2f}%")
    print(f"PDJ@0.50: {PDJ[3] * 100:.2f}%")
    return PDJ

def percentage_of_correct_keypoints(pred_json_path, gt_json_path, alphas=[0.05, 0.1, 0.2, 0.5], head_factor=1):
    """
    Calculate the percentage of correctly predicted keypoints.
    
    Parameters:
    pred_json_path (str): Path to the predicted JSON file.
    gt_json_path (str): Path to the ground truth JSON file.
    
    Returns:
    float: Percentage of correct keypoints.
    """
    pred_json = load_json_to_dataform(pred_json_path, norm=False)
    gt_json = load_json_to_dataform(gt_json_path, norm=False)
    
    PCK_005 = []
    PCK_010 = []
    PCK_015 = []
    PCK_020 = []
    PCK = [PCK_005, PCK_010, PCK_015, PCK_020]
    
    eyes_connection = (1, 2)
    
    for i in range(min(len(pred_json), len(gt_json))):
        pred_frame = pred_json[i]
        gt_frame = gt_json[i]
        gt_eye_keypoint1 = (gt_frame[eyes_connection[0] * 2], gt_frame[eyes_connection[0] * 2 + 1])
        gt_eye_keypoint2 = (gt_frame[eyes_connection[1] * 2], gt_frame[eyes_connection[1] * 2 + 1])
        current_head_size = calculate_limb_l2_distance(gt_eye_keypoint1, gt_eye_keypoint2) * head_factor
        for k in range(len(alphas)):
            for j in range(0, len(pred_frame), 2):
                
                pred_keypoint = (pred_frame[j], pred_frame[j + 1])
                gt_keypoint = (gt_frame[j], gt_frame[j + 1])
                if gt_keypoint == (0, 0):
                    # If the ground truth keypoint is (0, 0), we consider it as not detected
                    # PCK[k].append(0)
                    continue
                d_j = calculate_limb_l2_distance(pred_keypoint, gt_keypoint)
                # print(d_j)
                # print(f"Keypoint {j // 2}: Predicted: {pred_keypoint}, Ground Truth: {gt_keypoint}, Distance: {d_j:.2f}, Head Size: {current_head_size:.2f}")
                if d_j <= current_head_size * alphas[k]:
                    PCK[k].append(1)
                else:
                    PCK[k].append(0)
        
    PCK = [sum(p) / len(p) for p in PCK]
    print(f"PCK@0.05: {PCK[0] * 100:.2f}%")
    print(f"PCK@0.10: {PCK[1] * 100:.2f}%")
    print(f"PCK@0.20: {PCK[2] * 100:.2f}%") 
    print(f"PCK@0.50: {PCK[3] * 100:.2f}%")
    return PCK

def compute_bbox(keypoints, part_definition):
    # print(keypoints.shape)
    keypoints = np.array(keypoints)
    keypoints = keypoints.reshape(-1)
    keypoints = keypoints.reshape(-1, 2)  # Ensure keypoints are in shape (N, 2)
    # print(part_definition)
    points = [keypoints[part] for part in part_definition]
    if len(points) == 0:
        return None
    points = np.array(points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    area = w * h
    return math.sqrt(area)

def object_keypoint_similarity(pred_json_path, gt_json_path):
    """
    Calculate the Object Keypoint Similarity (OKS) metric.
    
    Parameters:
    pred_json_path (str): Path to the predicted JSON file.
    gt_json_path (str): Path to the ground truth JSON file.
    
    Returns:
    float: OKS value.
    """
    part_index = ["eyes", "ears", "shoulders", "elbows", "wrists", "hips", "knees", "ankles"]
    connection_dict = {
        'eyes': (1, 2),
        'ears': (3, 4),
        'shoulders': (5, 6),
        'elbows': (7, 8),
        'wrists': (9, 10),
        'hips': (11, 12),
        'knees': (13, 14),
        'ankles': (15, 16),
    }
    connection_dict_inverse = {
        1: "eyes",
        2: "eyes",
        3: "ears",
        4: "ears",
        5: "shoulders",
        6: "shoulders",
        7: "elbows",
        8: "elbows",
        9: "wrists",
        10: "wrists",
        11: "hips",
        12: "hips",
        13: "knees",
        14: "knees",
        15: "ankles",
        16: "ankles",
    }
    factor_dict = {
        'eyes': 0.025,
        'ears': 0.035,
        'shoulders': 0.079,
        'elbows': 0.072,
        'wrists': 0.062,
        'hips': 0.107,
        'knees': 0.087,
        'ankles': 0.089,
    }
    pred_json = load_json_to_dataform(pred_json_path, norm=False)
    gt_json = load_json_to_dataform(gt_json_path, norm=False)
    
    OKS = 0
    total_len = 0
    
    for pred_frame, gt_frame in zip(pred_json, gt_json):
        
        
        for i in range(1, 17):
            pred_keypoint = (pred_frame[i * 2], pred_frame[i * 2 + 1])
            gt_keypoint = (gt_frame[i * 2], gt_frame[i * 2 + 1])
            if gt_keypoint == (0, 0):
                # If the ground truth keypoint is (0, 0), we consider it as not detected
                continue
            d_j = calculate_limb_l2_distance(pred_keypoint, gt_keypoint)
            sigma = compute_bbox(gt_frame, connection_dict[connection_dict_inverse[i]]) * factor_dict[connection_dict_inverse[i]]
            if sigma == 0:
                continue
            oks = np.exp(-(d_j ** 2) / (2 * sigma ** 2))
            OKS += oks
            total_len += 1
        # for j, pred_keypoint, gt_keypoint in enumerate(zip(pred_keypoints, gt_keypoints)):
            
        #     d_j = calculate_limb_l2_distance(pred_keypoint, gt_keypoint)
        #     current_part = connection_dict_inverse[j // 2 if j % 2 == 0 else j // 2 + 1]
        #     sigma = compute_bbox(pred_keypoint, connection_dict[current_part]) * factor_dict[current_part]
            
        #     if sigma == 0:
        #         continue
        #     oks = np.exp(-(d_j ** 2) / (2 * sigma ** 2))
        #     OKS += oks
            
    OKS /= total_len
    print(f"OKS: {OKS * 100:.2f}%")
    return OKS
    
def evaluate(pred_json_path, gt_json_path, skeleton_connection=None):
    """
    Evaluate the predicted keypoints against the ground truth.
    
    Parameters:
    pred_json_path (str): Path to the predicted JSON file.
    gt_json_path (str): Path to the ground truth JSON file.
    skeleton_connection (list): List of tuples representing connections between keypoints.
    
    Returns:
    dict: Evaluation metrics including PCP, PDJ, PCK, and OKS.
    """
    
    pcp = percentage_of_correct_parts(pred_json_path, gt_json_path, skeleton_connection)
    pdj = percent_of_detected_joints(pred_json_path, gt_json_path)
    pck = percentage_of_correct_keypoints(pred_json_path, gt_json_path)
    oks = object_keypoint_similarity(pred_json_path, gt_json_path)
    
    return {
        'PCP': pcp,
        'PDJ': pdj,
        'PCK': pck,
        'OKS': oks
    }
        
            
def evaluate_json_folder(pred_json_folder, gt_json_folder, skeleton_connection=None):
    """
    Evaluate all JSON files in the given folders.
    
    Parameters:
    pred_json_folder (str): Path to the folder containing predicted JSON files.
    gt_json_folder (str): Path to the folder containing ground truth JSON files.
    skeleton_connection (list): List of tuples representing connections between keypoints.
    
    Returns:
    dict: Evaluation metrics for each file.
    """
    pred_json_files = list(Path(pred_json_folder).glob("*.json"))
    gt_json_files = list(Path(gt_json_folder).glob("*.json"))
    
    # sort the files to ensure they match
    pred_json_files = sorted(pred_json_files, key=lambda x: int(x.name.split('-')[0].split('_')[1]))  # Sort by video index
    gt_json_files = sorted(gt_json_files, key=lambda x: int(x.name.split('_')[1].split('-')[0]))  # Sort by video index
    
    all_metrics = {}
    
    for pred_file, gt_file in zip(pred_json_files, gt_json_files):
        pred_file_path = str(pred_file)
        gt_file_path = str(gt_file)
        
        print(f"Evaluating {pred_file_path} against {gt_file_path}")
        metrics = evaluate(pred_file_path, gt_file_path, skeleton_connection)
        all_metrics[pred_file.name] = metrics
    
    return all_metrics

def find_best_metrics(metrics_dict):
    """
    Find the best metrics from the evaluation results.
    
    Parameters:
    metrics_dict (dict): Dictionary containing evaluation metrics for each file.
    
    Returns:
    dict: Best metrics found in the evaluation results.
    """
    best_metrics = {
        'PCP': 0,
        'PDJ': [0, 0, 0, 0],
        'PCK': [0, 0, 0, 0],
        'OKS': 0
    }
    
    for file_name, metrics in metrics_dict.items():
        if metrics['PCP'] > best_metrics['PCP']:
            best_metrics['PCP'] = metrics['PCP']
        for i in range(len(best_metrics['PDJ'])):
            if metrics['PDJ'][i] > best_metrics['PDJ'][i]:
                best_metrics['PDJ'][i] = metrics['PDJ'][i]
        for i in range(len(best_metrics['PCK'])):
            if metrics['PCK'][i] > best_metrics['PCK'][i]:
                best_metrics['PCK'][i] = metrics['PCK'][i]
        if metrics['OKS'] > best_metrics['OKS']:
            best_metrics['OKS'] = metrics['OKS']
    
    # print best metrics
    print(f"Best PCP: {best_metrics['PCP'] * 100:.2f}%")
    print(f"Best PDJ: {best_metrics['PDJ']}")
    print(f"Best PCK: {best_metrics['PCK']}")
    print(f"Best OKS: {best_metrics['OKS'] * 100:.2f}%")
    
    return best_metrics

def get_mean_metrics(metrics_dict):
    """
    Calculate the mean metrics from the evaluation results.
    
    Parameters:
    metrics_dict (dict): Dictionary containing evaluation metrics for each file.
    
    Returns:
    dict: Mean metrics calculated from the evaluation results.
    """
    mean_metrics = {
        'PCP': 0,
        'PDJ': [0, 0, 0, 0],
        'PCK': [0, 0, 0, 0],
        'OKS': 0
    }
    
    num_files = len(metrics_dict)
    
    for file_name, metrics in metrics_dict.items():
        mean_metrics['PCP'] += metrics['PCP']
        for i in range(len(mean_metrics['PDJ'])):
            mean_metrics['PDJ'][i] += metrics['PDJ'][i]
        for i in range(len(mean_metrics['PCK'])):
            mean_metrics['PCK'][i] += metrics['PCK'][i]
        mean_metrics['OKS'] += metrics['OKS']
    
    mean_metrics['PCP'] /= num_files
    mean_metrics['PDJ'] = [x / num_files for x in mean_metrics['PDJ']]
    mean_metrics['PCK'] = [x / num_files for x in mean_metrics['PCK']]
    mean_metrics['OKS'] /= num_files
    
    print(f"Mean PCP: {mean_metrics['PCP'] * 100:.2f}%")
    print(f"Mean PDJ: {mean_metrics['PDJ']}")
    print(f"Mean PCK: {mean_metrics['PCK']}")
    print(f"Mean OKS: {mean_metrics['OKS'] * 100:.2f}%")
    
    return mean_metrics
        
    
    