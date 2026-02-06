import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperDKLM(nn.Module):
    def __init__(self, num_clusters, alpha=1.0, beta=1.0, gamma=0.1, max_iter=10, device='cuda'):
        """
        [Pure Kernel Mapping DKLM]
        严格遵循核空间映射逻辑：
        1. RBF 初始化 (几何先验)
        2. 谱流形优化 (流形正则化)
        3. 无任何人工缩放或强制归一化 Trick
        """
        super(HyperDKLM, self).__init__()
        self.k = num_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.device = device
        
        # RBF 初始带宽 (保持 0.5，这是高斯核的标准设置)
        self.init_gamma = 0.5 

    def _update_Z(self, K, C):
        """
        求解 Z = argmin ||Z - K|| + ...
        严格按照闭式解计算，不进行任何人为的归一化操作。
        """
        N = K.shape[0]
        I = torch.eye(N, device=self.device)
        
        # 求解线性系统 (alpha*K + beta*I) * Z = (alpha*K + beta*C)
        left = self.alpha * K + self.beta * I + 1e-5 * I # 极小的 jitter 仅防奇异
        right = self.alpha * K + self.beta * C
        
        try:
            Z_new = torch.linalg.solve(left, right)
        except RuntimeError:
            Z_new = torch.matmul(torch.linalg.pinv(left), right)
        
        # 仅保留非负约束 (因为相似度不能为负)
        Z_new = torch.relu(Z_new)
        
        # 去除自环 (我们只关心与他人的关系)
        Z_new.fill_diagonal_(0)
        
        # [重点] 绝对不要做 F.normalize(p=1)！
        # 如果做了，所有任务的权重和都一样，就体现不出"谁更重要"了。
        
        return Z_new

    def _update_S(self, C):
        """
        谱约束: 寻找前 k 个最小切图
        """
        N = C.shape[0]
        degree = torch.sum(C, dim=1)
        L_C = torch.diag(degree) - C
        L_C_sym = (L_C + L_C.t()) / 2
        
        try:
            eig_vals, eig_vecs = torch.linalg.eigh(L_C_sym)
        except RuntimeError:
            return torch.eye(N, device=self.device)[:, :self.k] @ torch.eye(N, device=self.device)[:, :self.k].t()

        U = eig_vecs[:, :self.k] 
        S_new = torch.matmul(U, U.t())
        return S_new

    def _update_C(self, Z, S):
        """
        辅助变量 C: 引入稀疏惩罚 gamma
        """
        N = Z.shape[0]
        diag_S = torch.diag(S).unsqueeze(1)
        ones = torch.ones(1, N, device=self.device)
        
        term_S = diag_S @ ones - S
        # 核心公式：Q = Z - (gamma/beta) * (D_s - S)
        Q = Z - (self.gamma / self.beta) * term_S
        
        Q = (Q + Q.t()) / 2
        
        # ReLU 相当于 Soft Thresholding，负责切断弱连接
        C_new = torch.relu(Q)
        C_new.fill_diagonal_(0)
        return C_new

    def _update_K(self, Z):
        """
        核学习: 反向更新 Kernel
        """
        W = (Z + Z.t()) / 2
        # 这里必须归一化，因为 Kernel 定义就是归一化的相似度，否则迭代会发散
        degree = W.sum(dim=1, keepdim=True) + 1e-8
        W_norm = W / torch.sqrt(degree @ degree.t())
        return W_norm

    def forward(self, features):
        """
        Returns:
            Z (Tensor): 纯粹的流形相似度矩阵
        """
        features = features.to(self.device)
        
        # 1. 强制归一化输入 (这是数学计算 RBF 的前提，不是 Trick)
        features = F.normalize(features, p=2, dim=1)
        
        N, D = features.shape
        
        # 2. RBF 初始化 (基于欧氏距离的核空间映射)
        dist_mat = torch.cdist(features, features, p=2).pow(2)
        K = torch.exp(-self.init_gamma * dist_mat)
        
        Z = K.clone().fill_diagonal_(0)
        C = Z.clone()
        
        # 3. 迭代优化 (在流形上寻找最优结构)
        with torch.no_grad():
            for i in range(self.max_iter):
                Z = self._update_Z(K, C)
                S = self._update_S(C)
                C = self._update_C(Z, S)
                K = self._update_K(Z)
        return Z