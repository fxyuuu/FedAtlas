import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FedAtlasSemCom:
    def __init__(self, device, args):
        self.device = device
        
        # PSR 参数 (模型压缩)
        self.sparsity_ratio = args.semcom_sparsity  # Top-S 比例 (e.g., 0.1)
        self.quant_bits = args.semcom_quant_bits    # 量化比特数 (e.g., 8)
        self.residual_error_w = {}                  # 累积的参数误差 m^(r-1)
        
        # DSR 参数 (特征压缩)
        self.use_dsr = args.semcom_dsr              # 开关
        self.codebook_size = args.semcom_codebook_size # Codebook 大小 M
        self.feature_dim = None                     # 特征维度
        self.codebook = {}                          # 客户端专属码本 B_i
        self.residual_error_f = {}                  # 累积的特征误差 e^(r-1)
        self.ema_alpha = 0.1                        # 码本更新平滑因子

    # =========================================================================
    # 1. Parameter Semantic Representation (PSR) - 对应论文 Eq. 16-21
    # =========================================================================
    def compress_model_update(self, client_idx, param_name, raw_update):
        """
        输入: 原始更新量 (Delta Theta)
        输出: 语义压缩后的更新量 (Reconstructed Update)
        """
        key = f"{client_idx}_{param_name}"
        
        # 1. 误差补偿 Eq. 16: u = update + m_(r-1)
        if key not in self.residual_error_w:
            self.residual_error_w[key] = torch.zeros_like(raw_update)
        
        compensated_update = raw_update + self.residual_error_w[key]
        
        # 展平
        flat_u = compensated_update.view(-1)
        numel = flat_u.numel()
        k = int(numel * self.sparsity_ratio)
        if k == 0: k = 1

        # 2. Top-S 稀疏化 Eq. 17-18
        # 选取绝对值最大的 Top-S 个元素
        values, indices = torch.topk(torch.abs(flat_u), k)
        # 获取原始符号的值
        top_k_values = flat_u[indices]
        
        # 3. 量化 Eq. 19-20
        # 简单均匀量化模拟
        L, R = top_k_values.min(), top_k_values.max()
        scale = (2 ** self.quant_bits - 1) / (R - L + 1e-8)
        
        # Quantize
        q_vals = torch.floor((top_k_values - L) * scale)
        # Dequantize (Inverse)
        rec_values = L + q_vals * (1.0 / scale)
        
        # 4. 重建稀疏向量
        reconstructed_flat = torch.zeros_like(flat_u)
        reconstructed_flat[indices] = rec_values
        reconstructed_update = reconstructed_flat.view(raw_update.shape)
        
        # 5. 更新残差 Eq. 21: m_(r) = u - u_hat
        self.residual_error_w[key] = compensated_update - reconstructed_update
        
        return reconstructed_update

    # =========================================================================
    # 2. Data Semantic Representation (DSR) - 对应论文 Eq. 23-27
    # =========================================================================
    def compress_task_feature(self, client_idx, task_name, raw_feature):
        """
        输入: 原始任务特征 (Phi)
        输出: 重建特征 (Phi_hat)
        """
        if not self.use_dsr:
            return raw_feature

        key = f"{client_idx}_{task_name}"
        
        # 初始化 Codebook (K-Means) - 仅在第一次运行时
        if key not in self.codebook:
            # 简单的随机选择作为初始化，或者运行简易 K-Means
            # 这里为了速度，从 raw_feature 中随机选 M 个作为初始聚类中心
            # 假设 raw_feature 是 [Dim] 或 [Batch, Dim]
            # 我们需要 feature 是 [Dim] 向量，或者是 Batch 的平均
            feat_vec = raw_feature.view(1, -1) if raw_feature.dim() == 1 else raw_feature
            
            # 如果维度不匹配，重置
            if self.feature_dim is None:
                self.feature_dim = feat_vec.shape[-1]
            
            # 随机初始化码本
            init_indices = torch.randperm(feat_vec.size(0))[:self.codebook_size]
            if feat_vec.size(0) < self.codebook_size:
                # 样本不够，填补随机数
                self.codebook[key] = torch.randn(self.codebook_size, self.feature_dim).to(self.device)
            else:
                self.codebook[key] = feat_vec[init_indices].detach().clone()
            
            self.residual_error_f[key] = torch.zeros_like(raw_feature)

        # 1. 误差补偿 Eq. 23: h = phi + e_(r-1)
        compensated_feat = raw_feature + self.residual_error_f[key]
        
        # 2. 向量量化 (VQ) - 寻找最近的码本索引 Eq. 24
        # compensated: [D], codebook: [M, D]
        # 计算距离
        dists = torch.cdist(compensated_feat.unsqueeze(0), self.codebook[key].unsqueeze(0)).squeeze(0) # [1, M]
        min_dist_idx = torch.argmin(dists)
        
        # 3. 重建特征 Eq. 25的前半部分
        reconstructed_feat = self.codebook[key][min_dist_idx].clone()
        
        # 4. 更新残差 Eq. 25: e_(r) = h - h_hat
        self.residual_error_f[key] = compensated_feat - reconstructed_feat
        
        # 5. 码本自适应更新 (EMA) Eq. 27
        # 论文中提到只有当 Error > Threshold 时才触发更新并上传
        # 这里我们模拟服务端接收到新中心后的 EMA 更新
        # new_center = compensated_feat (近似当前最佳代表)
        old_center = self.codebook[key][min_dist_idx]
        new_center = (1 - self.ema_alpha) * old_center + self.ema_alpha * compensated_feat.detach()
        self.codebook[key][min_dist_idx] = new_center
        
        return reconstructed_feat