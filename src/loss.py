import torch
import torch.nn as nn
import torch.nn.functional as F

# def vae_loss(recon_x, x, mu, logvar, beta=0.001, confidence_thresh=0.1):
#     """
#     recon_x: (B, T, D)
#     x:       (B, T, D)
#     D should be 51 = 17 keypoints × 3 (x, y, conf)
#     """
#     B, T, D = x.shape
#     device = x.device

#     confidence = x[:, :, 2::3]  # shape: (B, T, 17)

#     mask = (confidence >= confidence_thresh).float().unsqueeze(-1)  # (B, T, 17, 1)

#     keypoint_mask = mask.repeat(1, 1, 1, 3).reshape(B, T, D)

#     mse = (keypoint_mask * (recon_x - x) ** 2).sum() / keypoint_mask.sum().clamp(min=1.0)

#     kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

#     return mse + beta * kl


def vae_loss(predicted_next_frame, target_next_frame, mu, logvar, beta=0.001, confidence_thresh=0.1, lambda_smooth=0.1):
    # 參數名稱已根據新目標進行調整，以增加可讀性
    # predicted_next_frame: 模型預測的下一幀 (形狀: B, 1, D_full)
    # target_next_frame: 真實的下一幀 (形狀: B, 1, D_full)
    # mu, logvar: 從輸入序列 (window_size 幀) 編碼而來，形狀為 (B, latent_dim)
    
    B, T_frame, D_full = target_next_frame.shape # T_frame 在這裡將始終是 1 (因為是單幀)
    num_kpts = D_full // 3
    device = target_next_frame.device

    # 重塑張量以便處理每個關鍵點的 x, y, c
    target_reshaped = target_next_frame.view(B, T_frame, num_kpts, 3) # 形狀: (B, 1, num_kpts, 3)
    predicted_reshaped = predicted_next_frame.view(B, T_frame, num_kpts, 3) # 形狀: (B, 1, num_kpts, 3)

    true_xy = target_reshaped[..., :2]       # 真實的 x, y 座標
    recon_xy = predicted_reshaped[..., :2]    # 預測的 x, y 座標
    confidence = target_reshaped[..., 2]      # 使用目標幀的置信度進行加權

    # 重建損失 (Reconstruction Loss)
    # 計算預測幀與目標幀之間的 L1 平滑損失，並根據置信度加權
    weights = (confidence >= confidence_thresh).float()
    weights_exp = weights.unsqueeze(-1).expand_as(true_xy) # 擴展權重以匹配 xy 維度

    recon_loss_per_element = F.smooth_l1_loss(recon_xy, true_xy, reduction='none')
    weighted_recon_loss_sum = (recon_loss_per_element * weights_exp).sum()
    total_active_elements = weights_exp.sum().clamp(min=1.0) # 防止除以零
    recon_loss_val = weighted_recon_loss_sum / total_active_elements

    # KL 散度損失 (KL Divergence Loss)
    # 這個損失與 Encoder 相關，mu 和 logvar 是從整個輸入序列 (window_size 幀) 中得出的
    # 因此，正規化應該是除以批次大小 B
    kl_loss_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B # <--- 這裡已修正

    # 時間平滑損失 (Temporal Smoothness Loss)
    # 在單幀預測的場景下，T_frame 將始終為 1，因此此處定義的平滑損失將不起作用 (始終為 0)
    # 如果需要，可以重新設計一個新的平滑損失，例如比較輸入序列最後一幀與預測幀之間的平滑性
    if T_frame > 1: # 此條件將永遠為 False
        # 這部分邏輯已不適用於單幀預測，但為保持原函數結構，暫保留
        pred_diff = recon_xy[:, 1:] - recon_xy[:, :-1]
        true_diff = true_xy[:, 1:] - true_xy[:, :-1]
        smooth_loss = F.smooth_l1_loss(pred_diff, true_diff, reduction='mean')
    else:
        smooth_loss = torch.tensor(0.0, device=device) # 始終為 0

    # 總損失
    total_loss = recon_loss_val + beta * kl_loss_val + lambda_smooth * smooth_loss
    return total_loss, recon_loss_val, kl_loss_val, smooth_loss