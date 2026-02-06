import time
import copy
import torch
import random

def get_model_size(model):
    """计算模型参数大小 (MB)"""
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.numel() * 4  # float32 占 4 字节
        param_sum += param.numel()
    
    size_mb = param_size / 1024 / 1024
    return size_mb

def estimate_comm_time(model_size_mb, bandwidth_mbps, latency=0.0):
    """估算通信时间"""
    if bandwidth_mbps <= 0:
        return 0 
    
    transfer_time = model_size_mb / bandwidth_mbps
    total_time = transfer_time + latency
    return total_time

def apply_loss_to_state_dict(state_dict, loss_rate, device):
    """
    模拟丢包
    [修改] 增加对 'dklm_feat' 的保护，不进行丢包模拟
    """
    if loss_rate <= 0:
        return state_dict
    
    corrupted_dict = copy.deepcopy(state_dict)
    for key in corrupted_dict:
        # [保护] 跳过 DKLM 特征，确保语义向量完整传输
        if "dklm_feat" in key:
            continue

        param = corrupted_dict[key]
        if param.is_floating_point():
            # 生成伯努利掩码：1代表保留，0代表丢失
            mask = torch.bernoulli(torch.full_like(param, 1 - loss_rate, device=device))
            param.mul_(mask)
    return corrupted_dict

# [此前缺失的函数]
def get_parameter_dimensions(model):
    """
    获取模型参数维度详情
    Returns: 
        total_params (int): 总参数量
        dims_str (str): 参数形状字典的字符串表示
    """
    dims = {}
    total_params = 0
    # 使用 state_dict 遍历可以包含 buffer 和 parameter
    for name, param in model.state_dict().items():
        shape = list(param.size())
        dims[name] = shape
        total_params += param.numel()
    return total_params, str(dims)