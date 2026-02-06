import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperDKLM(nn.Module):
    def __init__(self, num_clusters, alpha=2.0, beta=1.0, gamma=0.1, max_iter=10, device='cuda'):
        """
        FedAtlas 核心模块：基于 DKLM 学习全局任务关系图谱 (Z 矩阵)
        Args:
            num_clusters (int): 预期的簇数量 (对应论文中的 k)
            alpha (float): 局部流形保持权重
            beta (float): 辅助矩阵松弛权重
            gamma (float): 块对角正则化权重
            max_iter (int): 优化迭代次数
        """
        super(HyperDKLM, self).__init__()
        self.k = num_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.device = device

    def _update_Z(self, K, C):
        # 更新自表示矩阵 Z
        # Z = (K + beta * I)^(-1) * (alpha * K + beta * C)
        N = K.shape[0]
        I = torch.eye(N, device=self.device)
        
        # 添加微小抖动 (1e-6) 保证矩阵可逆
        term_inv = torch.linalg.inv(K + self.beta * I + 1e-6 * I)
        term_right = self.alpha * K + self.beta * C
        
        Z_new = torch.matmul(term_inv, term_right)
        
        # 约束 Z >= 0 且 diag(Z) = 0 (避免自环)
        Z_new = torch.relu(Z_new)
        Z_new.fill_diagonal_(0)
        return Z_new

    def _update_S(self, C):
        # 更新 S 以引入块对角约束 (Laplacian 矩阵特征分解)
        N = C.shape[0]
        degree = torch.sum(C, dim=1)
        L_C = torch.diag(degree) - C
        
        # 对称化以保证数值稳定
        L_C_sym = (L_C + L_C.t()) / 2
        try:
            # eigh 用于对称矩阵，返回特征值(升序)和特征向量
            eig_vals, eig_vecs = torch.linalg.eigh(L_C_sym)
        except RuntimeError:
            # 极少数情况分解不收敛，返回默认投影
            return torch.eye(N, device=self.device)[:, :self.k] @ torch.eye(N, device=self.device)[:, :self.k].t()

        # 取前 k 个最小特征值对应的特征向量
        U = eig_vecs[:, :self.k] 
        S_new = torch.matmul(U, U.t())
        return S_new

    def _update_C(self, Z, S):
        # 更新辅助矩阵 C
        N = Z.shape[0]
        diag_S = torch.diag(S).unsqueeze(1)
        ones = torch.ones(1, N, device=self.device)
        
        term_S = diag_S @ ones - S
        A = Z - (self.gamma / self.beta) * term_S
        
        # A_hat = A - Diag(diag(A))
        diag_A = torch.diag(torch.diag(A))
        A_hat = A - diag_A
        
        # C = ReLU((A_hat + A_hat.T)/2)
        C_new = torch.relu((A_hat + A_hat.t()) / 2)
        return C_new

    def _update_K(self, Z):
        # 自适应更新核矩阵 K (基于乘性三角不等式)
        W = (Z + Z.t()) / 2
        degree = torch.sum(W, dim=1) + 1e-8
        d_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        
        G = torch.matmul(torch.matmul(d_inv_sqrt, W), d_inv_sqrt)
        max_G = torch.max(G)
        
        # 学习到的核矩阵
        K_new = torch.exp(-2 * max_G + G)
        return K_new

    def forward(self, features):
        """
        Args:
            features (Tensor): (N_total_tasks, D_dim) 输入特征矩阵
        Returns:
            Z (Tensor): (N_total_tasks, N_total_tasks) 学习到的全局关系矩阵
        """
        features = features.to(self.device)
        N, D = features.shape
        
        # 1. 初始化 Kernel (使用线性核初始化)
        features_norm = F.normalize(features, p=2, dim=1)
        K = torch.matmul(features_norm, features_norm.t())
        
        Z = torch.zeros(N, N, device=self.device)
        C = torch.zeros(N, N, device=self.device)
        
        # 2. DKLM 交替优化 (无需梯度)
        with torch.no_grad():
            for i in range(self.max_iter):
                Z = self._update_Z(K, C)
                S = self._update_S(C)
                C = self._update_C(Z, S)
                K = self._update_K(Z)
        
        # 返回原始 Z，在 aggregate 中根据需要进行归一化处理
        return Z