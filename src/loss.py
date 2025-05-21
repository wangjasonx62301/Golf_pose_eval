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


def vae_loss(recon_x, x, mu, logvar, beta=0.001, confidence_thresh=0.1):
    """
    計算 Time Series VAE 的加權損失函數。
    重構損失僅針對關鍵點的 (x, y) 座標，並根據置信度進行加權。

    Args:
        recon_x (torch.Tensor): 模型重構的輸出。形狀為 (B, T, D_full)。
                                D_full 應為 17 * 3 = 51。
                                包含 (x, y, c) 的重構。
        x (torch.Tensor): 原始輸入數據。形狀為 (B, T, D_full)。
                          D_full 應為 17 * 3 = 51。
                          包含真實的 (x, y, c)。
        mu (torch.Tensor): 潛在空間的均值。形狀為 (B, latent_dim)。
        logvar (torch.Tensor): 潛在空間方差的對數。形狀為 (B, latent_dim)。
        beta (float): KL 散度的權重，用於平衡重構損失和 KL 散度。
        confidence_thresh (float): 用於創建遮罩的置信度閾值。低於此閾值的關鍵點將具有較低的重構損失權重。
    
    Returns:
        tuple: (total_loss, recon_loss_val, kl_loss_val)
               total_loss: 總損失。
               recon_loss_val: 重構損失的平均值。
               kl_loss_val: KL 散度的平均值。
    """
    B, T, D_full = x.shape # D_full = num_keypoints * 3 (x, y, c)
    num_kpts = D_full // 3
    device = x.device

    # 1. 提取原始 x, y 座標 和 置信度
    # 將輸入張量重新塑形為 (B, T, num_kpts, 3) 以便於提取
    x_reshaped = x.reshape(B, T, num_kpts, 3)
    recon_x_reshaped = recon_x.reshape(B, T, num_kpts, 3)

    # 提取真實的 x, y 座標
    true_xy = x_reshaped[:, :, :, :2] # shape: (B, T, num_kpts, 2)
    # 提取重構的 x, y 座標
    recon_xy = recon_x_reshaped[:, :, :, :2] # shape: (B, T, num_kpts, 2)

    # 提取置信度 (用於權重計算)
    confidence = x_reshaped[:, :, :, 2] # shape: (B, T, num_kpts)

    # 2. 創建權重/遮罩
    # 將置信度轉換為浮點數，並根據閾值創建二元遮罩 (0 或 1)
    # 你在預處理中已經將插值點的置信度設為 0.01
    # 如果 confidence_thresh 設為 0.05 (例如)，那麼這些 0.01 的點就會被設為 0
    weights = (confidence >= confidence_thresh).float() # shape: (B, T, num_kpts)
    
    # 擴展權重維度，使其與 true_xy 和 recon_xy 的 shape 匹配 (為每個 x,y 分量賦予相同權重)
    weights_expanded = weights.unsqueeze(-1).expand_as(true_xy) # shape: (B, T, num_kpts, 2)

    # 3. 計算加權重構損失 (推薦使用 Huber Loss / Smooth L1 Loss)
    # reduction='none' 保留每個元素的損失
    # 重構損失只針對 (x, y) 座標
    # recon_loss_per_element = F.mse_loss(recon_xy, true_xy, reduction='none') # (B, T, num_kpts, 2)
    recon_loss_per_element = F.smooth_l1_loss(recon_xy, true_xy, reduction='none') # (B, T, num_kpts, 2)

    # 將損失與權重相乘，然後求和並除以有效權重的總和 (避免除以零)
    weighted_recon_loss = (recon_loss_per_element * weights_expanded).sum()

    # 計算有效權重的總和，用於正規化損失。
    # 確保 `total_active_elements` 至少為 1，以避免除以零。
    total_active_elements = weights_expanded.sum().clamp(min=1.0)
    
    # 正規化重構損失：將加權損失總和除以有效元素的總數
    # 這樣得到的 recon_loss_val 才是每個有效元素的平均損失
    recon_loss_val = weighted_recon_loss / total_active_elements

    # 4. 計算 KL 散度損失 (與你原來的邏輯相同)
    # 這裡的 logvar 和 mu 是從編碼器輸出，形狀通常是 (B, latent_dim)
    kl_loss_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 對 KL 散度進行批次平均 (或你需要的其他正規化)
    # 如果你希望 KL 散度也與時間步長 T 相關，可以除以 B * T
    # 但通常只除以 B 比較常見，表示每個樣本的平均 KL 散度
    kl_loss_val /= B * T

    # 5. 總損失
    total_loss = recon_loss_val + beta * kl_loss_val

    return total_loss, recon_loss_val, kl_loss_val
