import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from utils import move_ckpt
from models.dklm import HyperDKLM

# [新增] 引入绘图和文件操作库
import os
import matplotlib
# 设置后端防止在无屏幕服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

def get_encoder_keys(all_keys):
    """
    Get keys of encoder parameters
    """
    return list(filter(lambda x: 'backbone' in x, all_keys))

def get_decoder_keys(all_keys):
    """
    Get keys of decoder parameters
    """
    return list(filter(lambda x: 'decoders' in x, all_keys))

def get_model_soup(param_dict_list):
    """
    Get the average of parameters in list
    """
    soup_param_dict = {}
    layers = param_dict_list[0].keys()
    for layer in layers:
        soup_param_dict[layer] = torch.mean(
            torch.stack(
                [param_dict_list[i][layer] for i in range(len(param_dict_list))]
            ),
            dim=0,
        )
    return soup_param_dict

def get_delta_dict_list(param_dict_list, last_param_dict_list):
    """
    Get the difference between current and last parameters
    """
    # a list of length N, each element is a dict of delta parameters
    delta_dict_list = []
    layers = param_dict_list[0].keys()
    for i in range(len(param_dict_list)):
        delta_dict_list.append({})
        for layer in layers:
            delta_dict_list[i][layer] = (
                param_dict_list[i][layer] - last_param_dict_list[i][layer]
            )
    return delta_dict_list

def get_encoder_params(all_clients, ckpt):
    """
    Get encoder parameters from checkpoint
    """
    all_name_keys = [
        name for name, _ in all_clients[0]['model'].module.named_parameters()
    ]
    encoder_keys = get_encoder_keys(all_name_keys)
    encoder_param_dict_list = []
    layers = []
    shapes = []

    for model_idx in range(len(ckpt)):
        param_dict = {}
        for key in encoder_keys:
            # key=prefix+'.'+layer
            prefix, layer = key.split('.', 1)
            param_dict[layer] = ckpt[model_idx][key]
        encoder_param_dict_list.append(param_dict)

    # Get layers and shapes (same for all encoders)
    for key in encoder_keys:
        layers.append(key.split('.', 1)[1])
        shapes.append(ckpt[0][key].shape)

    return encoder_param_dict_list, encoder_keys, layers, shapes

def get_decoder_params(all_clients, ckpt):
    """
    Get decoder parameters from checkpoint
    """
    N = len(all_clients)
    n_st = sum(
        [len(all_clients[i]['tasks']) == 1 for i in range(N)]
    )  # number of st clients
    K = sum([len(all_clients[i]['tasks']) for i in range(N)])  # number of decoders

    decoder_keys = []
    layers = []
    shapes = []

    for idx in range(N):
        all_name_keys = [
            key for key, _ in all_clients[idx]['model'].module.named_parameters()
        ]
        decoder_keys += get_decoder_keys(all_name_keys)
    decoder_keys = list(set(decoder_keys))

    decoder_param_dict_list = []
    decoders_prefix = []

    # st client decoders
    for model_idx in range(n_st):
        assert len(all_clients[model_idx]['tasks']) == 1
        param_dict = {}
        for key in decoder_keys:
            if key in ckpt[model_idx].keys():
                # key=prefix+'.'+layer
                prefix = (
                    key.split('.', 2)[0] + '.' + key.split('.', 2)[1]
                )  # decoders.task
                layer = key.split('.', 2)[2]
                param_dict[layer] = ckpt[model_idx][key]

                if model_idx == 0:
                    layers.append(layer)
                    shapes.append(ckpt[0][key].shape)

        decoders_prefix.append(prefix)
        decoder_param_dict_list.append(param_dict)

    # mt client decoders
    for model_idx in range(n_st, N):
        prefix_list = []  # decoder prefixs in one mt client
        for task in all_clients[model_idx]['tasks']:
            prefix_list.append('decoders.' + task)
        prefix_list = sorted((prefix_list))  # keep the order

        for i, prefix in enumerate(prefix_list):
            # Get each task-specific decoder
            param_dict = {}
            for key in decoder_keys:
                if key in ckpt[model_idx].keys() and prefix in key:
                    layer = key.split('.', 2)[2]
                    param_dict[layer] = ckpt[model_idx][key]

                    if model_idx == 0 and i == 0:
                        layers.append(layer)
                        shapes.append(ckpt[0][key].shape)

            decoder_param_dict_list.append(param_dict)
        decoders_prefix += prefix_list

    assert len(decoders_prefix) == K
    assert len(decoder_param_dict_list) == K

    return decoder_param_dict_list, decoders_prefix, decoder_keys, layers, shapes

def get_ca_delta(flatten_delta_list, alpha, rescale=1):
    """
    Solve for aggregated conflict-averse delta
    """
    N = len(flatten_delta_list)
    grads = torch.stack(flatten_delta_list).t()  # [d , N]
    GG = grads.t().mm(grads).cpu()  # [N, N]
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
            + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    ww = torch.Tensor(res.x).to(grads.device)
    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2)
    else:
        final_update = g / (1 + alpha)

    return final_update

def flatten_param(param_dict_list, layers):
    """
    Flatten a dict of parameters into a vector
    """
    flatten_list = [
        torch.cat([param_dict_list[idx][layer].flatten() for layer in layers])
        for idx in range(len(param_dict_list))
    ]
    assert len(flatten_list[0].shape) == 1
    return flatten_list

def unflatten_param(flatten_list, shapes, layers):
    """
    Unflatten a vector into a dict of parameters
    """
    param_dict_list = []
    for model_idx in range(len(flatten_list)):
        start = 0
        param_dict_list.append({})
        for layer, shape in zip(layers, shapes):
            end = start + np.prod(shape)
            param_dict_list[model_idx][layer] = flatten_list[model_idx][
                start:end
            ].reshape(shape)
            start = end
    return param_dict_list

def aggregate(
    all_clients,
    save_ckpt,
    last_ckpt,
    hyperweight=None,
    encoder_agg='none',
    decoder_agg='none',
    ca_c=0.4,
    save_dir=None,
    current_round=0,
    local_rank=0
):
    '''
    Main aggregation function
    '''
    assert len(all_clients) == len(save_ckpt)
    N = len(all_clients)
    n_st = sum([len(client['tasks']) == 1 for client in all_clients])
    n_mt_tasks = [
        len(all_clients[i]['tasks']) for i in range(n_st, N)
    ] 

    # =============================================================
    # [FedAtlas/DKLM] Logic: Learn Manifold Z
    # =============================================================
    Z_matrix = None
    row_mapping = [] 
    
    # Enable DKLM if either encoder or decoder uses it
    use_dklm = (encoder_agg == 'dklm') or (decoder_agg == 'dklm')
    
    if use_dklm and local_rank == 0:
        if hyperweight is None or 'dklm_module' not in hyperweight:
            print(f"[Warning] use_dklm=True but 'dklm_module' missing in hyperweight! Aggregation will degrade.")

    if use_dklm and hyperweight is not None and 'dklm_module' in hyperweight:
        dklm_module = hyperweight['dklm_module']
        all_feats = []
        
        for client_idx, client_ckpt in enumerate(save_ckpt):
            feat_keys = [k for k in client_ckpt.keys() if 'dklm_feat_' in k]
            
            for key in feat_keys:
                start_idx = key.find('dklm_feat_') + len('dklm_feat_')
                task_name = key[start_idx:]
                feat_vec = client_ckpt[key].float()
                all_feats.append(feat_vec)
                row_mapping.append({'client': client_idx, 'task': task_name})
        
        if local_rank == 0 and current_round > 0: 
             print(f"[DKLM Debug] Round {current_round}: Collected {len(all_feats)} feature vectors.")

        if len(all_feats) > 1:
            device = dklm_module.device
            
            # Step 1: Normalize
            X_feat = torch.stack(all_feats).to(device)
            X_feat = F.normalize(X_feat, p=2, dim=1)

            # Step 2: Project
            projected_feat = dklm_module(X_feat)

            # Step 3: Compute Z
            if projected_feat.shape == (len(all_feats), len(all_feats)):
                raw_z = projected_feat
            else:
                projected_feat = F.normalize(projected_feat, p=2, dim=1)
                raw_z = torch.mm(projected_feat, projected_feat.t())

            # Step 4: Temperature & Softmax
            temperature = 0.5 
            scaled_z = raw_z / temperature
            Z_matrix = F.softmax(scaled_z, dim=1)

            if torch.isnan(Z_matrix).any():
                print(f"[Warning] DKLM Z-Matrix broke at Round {current_round}. Fallback to Identity.")
                Z_matrix = torch.eye(len(all_feats)).to(device)

            # --- Save Logs & Plots ---
            if local_rank == 0 and save_dir is not None:
                dklm_log_dir = os.path.join(save_dir, "dklm_logs")
                os.makedirs(dklm_log_dir, exist_ok=True)
                
                z_np = Z_matrix.detach().cpu().numpy()
                np.save(os.path.join(dklm_log_dir, f"Z_matrix_r{current_round}.npy"), z_np)
                
                try:
                    plt.figure(figsize=(10, 8))
                    labels = [f"C{info['client']}_{info['task'][:3]}" for info in row_mapping]
                    sns.heatmap(z_np, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False)
                    plt.title(f"FedAtlas Relations (T={temperature}) R{current_round}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(dklm_log_dir, f"Z_heatmap_r{current_round}.png"))
                    plt.close('all')
                except Exception as e:
                    print(f"[Warning] Heatmap plot failed: {e}")
                
                with open(os.path.join(dklm_log_dir, f"mapping_r{current_round}.txt"), "w") as f:
                    for idx, info in enumerate(row_mapping):
                        f.write(f"{idx}: Client={info['client']}, Task={info['task']}\n")

        else:
            if use_dklm:
                Z_matrix = None

    if encoder_agg == 'none' and decoder_agg == 'none':
        return 

    update_ckpt = copy.deepcopy(save_ckpt) 

    encoder_param_list, encoder_keys, enc_layers, enc_shapes = get_encoder_params(
        all_clients, save_ckpt
    )

    # =============================================================
    # Encoder Aggregation Logic
    # =============================================================
    if encoder_agg == 'none':
        del encoder_param_list
        pass

    elif encoder_agg in ['fedavg'] or (encoder_agg == 'dklm' and Z_matrix is None):
        if encoder_agg == 'dklm':
             print(f"[Info] Round {current_round}: Z_matrix is None. Encoder Fallback to FedAvg.")
        
        new_encoder_param = get_model_soup(encoder_param_list)

        for model_idx in range(N):
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                update_ckpt[model_idx][key] = new_encoder_param[layer]

        del encoder_param_list, new_encoder_param

    elif encoder_agg == 'dklm' and Z_matrix is not None:
        # =================================================================
        # [SOTA 推荐] Personalized Graph Aggregation (个性化图聚合)
        # 核心：保留 Z 矩阵的完整结构，为每个客户端定制模型，拒绝"大锅饭"。
        # =================================================================
        
        # 1. 转换 Task-level Z -> Client-level Weights
        W_clients = torch.zeros(N, N).to(Z_matrix.device)
        for i in range(N):
            tasks_i = [idx for idx, info in enumerate(row_mapping) if info['client'] == i]
            for j in range(N):
                tasks_j = [idx for idx, info in enumerate(row_mapping) if info['client'] == j]
                if not tasks_i or not tasks_j: continue
                # 取均值，反映两个客户端之间的整体相似度
                W_clients[i, j] = Z_matrix[np.ix_(tasks_i, tasks_j)].abs().mean()

        # 2. 归一化 (Row Normalization)
        # 策略：给对角线(自己)加一个保底权重(比如 1.0)，防止被别人带偏
        # 这一步非常重要，能稳住训练！
        W_clients.fill_diagonal_(1.0) 
        W_clients = F.normalize(W_clients, p=1, dim=1)

        if local_rank == 0 and save_dir is not None:
             dklm_log_dir = os.path.join(save_dir, "dklm_logs")
             os.makedirs(dklm_log_dir, exist_ok=True)
             
             # 1. 保存纯矩阵 (方便后续画图)
             np.savetxt(os.path.join(dklm_log_dir, f"weights_enc_matrix_r{current_round}.txt"), W_clients.detach().cpu().numpy())
             
             # 2. 保存可读日志 (方便肉眼检查)
             enc_logs = []
             w_np = W_clients.detach().cpu().numpy()
             for i in range(N):
                 log_str = f"Target C{i} Backbone: "
                 details = []
                 for j in range(N):
                     val = w_np[i, j]
                     if val > 0.001: # 只记录有意义的权重
                         details.append(f"C{j}={val:.3f}")
                 log_str += ", ".join(details) + "\n"
                 enc_logs.append(log_str)
                 
             with open(os.path.join(dklm_log_dir, f"weights_enc_personalized_r{current_round}.txt"), "w") as f:
                 f.writelines(enc_logs)
        
        # 3. 个性化聚合 (双重循环)
        # 预先堆叠参数以加速
        stack_params = {}
        layers = encoder_param_list[0].keys()
        for layer in layers:
            stack_params[layer] = torch.stack([encoder_param_list[k][layer] for k in range(N)]).to(Z_matrix.device)

        # 为每个 Client i 聚合专属模型
        for target_i in range(N):
            weights = W_clients[target_i] # 第 i 行：别人对我的贡献度
            
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                W_all = stack_params[layer]
                
                # 调整形状广播: [N] -> [N, 1, 1...]
                view_shape = [-1] + [1] * (W_all.dim() - 1)
                
                # Sum( w_{i,j} * theta_j )
                personalized_param = (W_all * weights.view(view_shape)).sum(dim=0).cpu()
                update_ckpt[target_i][key] = personalized_param

        # 清理
        del encoder_param_list, stack_params
    # elif encoder_agg == 'dklm' and Z_matrix is not None:
    #     # =================================================================
    #     # [方案一] Global Manifold Centrality (全局流形中心性)
    #     # =================================================================
    #     client_scores = torch.zeros(N).to(Z_matrix.device)
        
    #     # 遍历每一列 (c_idx 代表作为"贡献者"的特征，属于客户端 j)
    #     for c_idx, info_j in enumerate(row_mapping):
    #         j = info_j['client'] 
            
    #         # 创建掩码：我们要计算 j 这一列的和，但要排除 j 自己对自己投票的行
    #         mask = torch.ones(Z_matrix.shape[0], dtype=torch.bool).to(Z_matrix.device)
            
    #         for r_idx, info_i in enumerate(row_mapping):
    #             # 只要行属于客户端 j 自己，就设为 False
    #             if info_i['client'] == j:
    #                 mask[r_idx] = False
            
    #         # 累加这一列中，来自所有"外人"（包括同行和跨行）的权重
    #         score = Z_matrix[mask, c_idx].abs().sum()
            
    #         # 累加到客户端 j 的总分上
    #         client_scores[j] += score
        
    #     # 归一化权重
    #     total_score = client_scores.sum() + 1e-8
    #     encoder_weights = client_scores / total_score
        
    #     # 保存 Encoder 权重日志
    #     if local_rank == 0 and save_dir is not None:
    #          dklm_log_dir = os.path.join(save_dir, "dklm_logs")
    #          os.makedirs(dklm_log_dir, exist_ok=True)
    #          np.savetxt(os.path.join(dklm_log_dir, f"weights_enc_r{current_round}.txt"), encoder_weights.cpu().numpy())

    # elif encoder_agg == 'dklm' and Z_matrix is not None:
    #     client_scores = torch.zeros(N).to(Z_matrix.device)
        
    #     for r_idx, info_i in enumerate(row_mapping):
    #         i = info_i['client']; t = info_i['task']
    #         valid_cols = []
    #         for c_idx, info_j in enumerate(row_mapping):
    #             if info_j['client'] != i and info_j['task'] != t:
    #                 valid_cols.append(c_idx)
    #         if valid_cols:
    #             score = Z_matrix[r_idx, valid_cols].abs().sum()
    #             client_scores[i] += score
        
        
    #     total_score = client_scores.sum() + 1e-8
    #     encoder_weights = client_scores / total_score
        
    #     if local_rank == 0 and save_dir is not None:
    #          dklm_log_dir = os.path.join(save_dir, "dklm_logs")
    #          # [关键修复] 强制创建目录，防止文件未找到错误
    #          os.makedirs(dklm_log_dir, exist_ok=True)
    # #          np.savetxt(os.path.join(dklm_log_dir, f"weights_enc_r{current_round}.txt"), encoder_weights.cpu().numpy())
        
    #     new_encoder_param = {}
    #     layers = encoder_param_list[0].keys()
    #     for layer in layers:
    #         params_stack = torch.stack([encoder_param_list[i][layer] for i in range(N)]).to(Z_matrix.device)
    #         view_shape = [-1] + [1] * (params_stack.dim() - 1)
    #         weights_view = encoder_weights.view(view_shape)
    #         new_encoder_param[layer] = (params_stack * weights_view).sum(dim=0).cpu()

    #     for model_idx in range(N):
    #         for key in encoder_keys:
    #             layer = key.split('.', 1)[1]
    #             update_ckpt[model_idx][key] = new_encoder_param[layer]
        
    #     del encoder_param_list, new_encoder_param

    elif encoder_agg in ['conflict_averse']:
        last_encoder_param_list, _, _, _ = get_encoder_params(all_clients, last_ckpt)
        encoder_delta_list = get_delta_dict_list(encoder_param_list, last_encoder_param_list)
        flatten_last_encoder = flatten_param(last_encoder_param_list, enc_layers)
        del last_encoder_param_list
        flatten_encoder_delta = flatten_param(encoder_delta_list, enc_layers)
        del encoder_delta_list
        flatten_delta_update = get_ca_delta(flatten_encoder_delta, ca_c) 
        group = [0]; homo_avg = flatten_encoder_delta[0]; i = 1
        while i < N:
            if (all_clients[i]['dataname'] == all_clients[i - 1]['dataname'] and all_clients[i]['tasks'] == all_clients[i - 1]['tasks']):
                homo_avg += flatten_encoder_delta[i]; group.append(i)
            else:
                homo_avg /= len(group); 
                for j in group: flatten_encoder_delta[j] = homo_avg
                group = [i]; homo_avg = flatten_encoder_delta[i]
            i += 1
        homo_avg /= len(group)
        for j in group: flatten_encoder_delta[j] = homo_avg
        assert hyperweight['enc'] is not None
        flatten_new_encoder = hyperweight['enc'](flatten_last_encoder, flatten_encoder_delta, flatten_delta_update)
        hyperweight['last_enc_output'] = flatten_new_encoder
        del flatten_last_encoder, flatten_encoder_delta, flatten_delta_update
        new_encoder_param_list = unflatten_param(flatten_new_encoder, enc_shapes, enc_layers)
        for model_idx in range(N):
            for key in encoder_keys:
                layer = key.split('.', 1)[1]
                update_ckpt[model_idx][key] = new_encoder_param_list[model_idx][layer]
        del new_encoder_param_list
    else:
        raise NotImplementedError

    # =============================================================
    # Decoder Aggregation Logic
    # =============================================================
    decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = get_decoder_params(all_clients, save_ckpt)

    if decoder_agg == 'none':
        del decoder_param_list
        pass

    elif decoder_agg in ['fedavg'] or (decoder_agg == 'dklm' and Z_matrix is None):
        if decoder_agg == 'dklm':
             print(f"[Info] Round {current_round}: Z_matrix is None. Decoder Fallback to FedAvg.")
        new_decoder_param = get_model_soup(decoder_param_list)
        for i, prefix in enumerate(decoders_prefix):
            if i >= n_st:
                model_idx = n_st + (i - n_st) // (n_mt_tasks[0])
            else:
                model_idx = i
            for layer in dec_layers:
                update_ckpt[model_idx][prefix + '.' + layer] = new_decoder_param[layer]
        del decoder_param_list, new_decoder_param

    elif decoder_agg == 'dklm' and Z_matrix is not None:
        # =================================================================
        # [SOTA 方案] Personalized Decoder Aggregation (真正千人千面)
        # 逻辑：为每一个客户端 i，单独聚合一个适合它的 Decoder
        # =================================================================
        
        # 1. 建立索引映射：快速找到参数在哪
        map_client_task_to_list_idx = {}
        for list_idx, prefix in enumerate(decoders_prefix):
            if list_idx >= n_st:
                mt_offset = list_idx - n_st; tasks_per_mt = n_mt_tasks[0]
                client_idx = n_st + (mt_offset // tasks_per_mt)
            else: client_idx = list_idx
            task_name = prefix.split('.')[-1]
            map_client_task_to_list_idx[(client_idx, task_name)] = list_idx

        # 2. 辅助函数：查 Z 索引
        def get_z_index(c_idx, t_name):
            for r, info in enumerate(row_mapping):
                if info['client'] == c_idx and info['task'] == t_name: return r
            return None

        # [新增 1/3] 初始化日志容器
        decoder_logs = []
        # 3. 双重循环：遍历目标客户端 -> 遍历盟友
        # 这才是个性化的关键：Target Client 在外层循环
        for target_client_idx in range(N):
            for task_name in all_clients[target_client_idx]['tasks']:
                
                # 目标参数位置
                target_list_idx = map_client_task_to_list_idx.get((target_client_idx, task_name))
                target_z_idx = get_z_index(target_client_idx, task_name)
                if target_list_idx is None or target_z_idx is None: continue

                # --- 寻找盟友 (Peers) ---
                peer_clients = []
                peer_weights = []
                
                for peer_client_idx in range(N):
                    # 必须做同一个任务
                    if task_name not in all_clients[peer_client_idx]['tasks']: continue
                    
                    peer_z_idx = get_z_index(peer_client_idx, task_name)
                    if peer_z_idx is None: continue
                    
                    # [核心] 取出点对点权重: target 对 peer 的认可度
                    w = Z_matrix[target_z_idx, peer_z_idx].item()
                    
                    # 策略：给自己加权 (Self-Attention Bias)
                    # 保证即使周围都是噪声，自己也能保留大部分知识
                    if peer_client_idx == target_client_idx:
                        w = 1.0 # 或者更大，如 2.0，视置信度而定
                        
                    peer_clients.append(peer_client_idx)
                    peer_weights.append(w)
                
                if not peer_clients: continue

                # 归一化权重 (Sum = 1)
                peer_weights = torch.tensor(peer_weights).to(Z_matrix.device)
                peer_weights = F.normalize(peer_weights, p=1, dim=0)

                # [新增 2/3] 收集权重日志 (仅在 Rank 0 进行，节省资源)
                if local_rank == 0 and save_dir is not None:
                    log_str = f"Target C{target_client_idx} [{task_name}]: "
                    details = []
                    for k, p_idx in enumerate(peer_clients):
                        # 只记录有贡献的 (权重 > 0.001)
                        if peer_weights[k] > 0.001:
                            details.append(f"C{p_idx}={peer_weights[k]:.3f}")
                    log_str += ", ".join(details) + "\n"
                    decoder_logs.append(log_str)
                
                # --- 定制化聚合 ---
                target_prefix = decoders_prefix[target_list_idx]
                for layer in dec_layers:
                    weighted_sum = 0
                    for k, peer_client_idx in enumerate(peer_clients):
                        w = peer_weights[k]
                        peer_list_idx = map_client_task_to_list_idx[(peer_client_idx, task_name)]
                        param = decoder_param_list[peer_list_idx][layer].to(Z_matrix.device)
                        weighted_sum += w * param
                        
                    # 赋值：这是专门为 target_client_idx 定制的参数
                    update_ckpt[target_client_idx][f"{target_prefix}.{layer}"] = weighted_sum.cpu()

        # [新增 3/3] 将日志写入文件
        if local_rank == 0 and save_dir is not None and decoder_logs:
            dklm_log_dir = os.path.join(save_dir, "dklm_logs")
            os.makedirs(dklm_log_dir, exist_ok=True)
            with open(os.path.join(dklm_log_dir, f"weights_dec_personalized_r{current_round}.txt"), "w") as f:
                f.writelines(decoder_logs)
        del decoder_param_list
    # elif decoder_agg == 'dklm' and Z_matrix is not None:
    #     all_tasks = set([info['task'] for info in row_mapping])
    #     decoder_weights_log = {}
        
    #     for task_name in all_tasks:
    #         client_indices_with_task = []
    #         row_indices_in_Z = []
            
    #         for r_idx, info in enumerate(row_mapping):
    #             if info['task'] == task_name:
    #                 client_indices_with_task.append(info['client'])
    #                 row_indices_in_Z.append(r_idx)
            
    #         if len(client_indices_with_task) < 2: continue
            
    #         task_scores = torch.zeros(len(client_indices_with_task)).to(Z_matrix.device)
    #         for k, (client_idx, r_idx) in enumerate(zip(client_indices_with_task, row_indices_in_Z)):
    #             score = 0
    #             for m, (other_client_idx, c_idx) in enumerate(zip(client_indices_with_task, row_indices_in_Z)):
    #                 if client_idx != other_client_idx:
    #                     score += Z_matrix[r_idx, c_idx].abs() + Z_matrix[c_idx, r_idx].abs()
    #             task_scores[k] = score
            
    #         total_task_score = task_scores.sum() + 1e-8
    #         decoder_weights = task_scores / total_task_score

    #         decoder_weights_log[task_name] = {
    #             'clients': client_indices_with_task,
    #             'weights': decoder_weights.cpu().numpy()
    #         }
            
    #         current_task_indices = [] 
    #         for i, prefix in enumerate(decoders_prefix):
    #             if prefix.endswith(f".{task_name}"):
    #                   if i >= n_st:
    #                       mt_offset = i - n_st; tasks_per_mt = n_mt_tasks[0]
    #                       model_idx = n_st + (mt_offset // tasks_per_mt)
    #                   else: model_idx = i
    #                   if model_idx in client_indices_with_task:
    #                       current_task_indices.append((i, model_idx))
            
    #         if not current_task_indices: continue
            
        #     new_task_decoder_param = {}
        #     for layer in dec_layers:
        #         params_tensor_list = []
        #         weights_tensor_list = []
        #         for list_idx, model_idx in current_task_indices:
        #             try:
        #                 w_idx = client_indices_with_task.index(model_idx)
        #                 w = decoder_weights[w_idx]
        #                 p = decoder_param_list[list_idx][layer].to(Z_matrix.device)
        #                 params_tensor_list.append(p)
        #                 weights_tensor_list.append(w)
        #             except ValueError: continue
        #         if not params_tensor_list: continue
        #         params_stack = torch.stack(params_tensor_list)
        #         weights_stack = torch.stack(weights_tensor_list)
        #         weights_stack = weights_stack / (weights_stack.sum() + 1e-8)
        #         view_shape = [-1] + [1] * (params_stack.dim() - 1)
        #         weights_view = weights_stack.view(view_shape)
        #         new_task_decoder_param[layer] = (params_stack * weights_view).sum(dim=0).cpu()

        #     for list_idx, model_idx in current_task_indices:
        #         prefix = decoders_prefix[list_idx]
        #         for layer in dec_layers:
        #             update_ckpt[model_idx][prefix + '.' + layer] = new_task_decoder_param[layer]

        # if local_rank == 0 and save_dir is not None and decoder_weights_log:
        #      dklm_log_dir = os.path.join(save_dir, "dklm_logs")
        #      # [关键修复] 强制创建目录，防止文件未找到错误
        #      os.makedirs(dklm_log_dir, exist_ok=True)
        #      with open(os.path.join(dklm_log_dir, f"weights_dec_r{current_round}.txt"), "w") as f:
        #          for t_name, data in decoder_weights_log.items():
        #              f.write(f"Task: {t_name}\n")
        #              for c_idx, w in zip(data['clients'], data['weights']):
        #                  f.write(f"  Client {c_idx}: {w:.4f}\n")

        # del decoder_param_list

    elif decoder_agg in ['cross_attention']:
        assert hyperweight['dec'] is not None
        last_decoder_param_list, _, _, _, _ = get_decoder_params(all_clients, last_ckpt)
        decoder_delta_list = get_delta_dict_list(decoder_param_list, last_decoder_param_list)
        new_decoder_param_list = hyperweight['dec'](last_decoder_param_list, decoder_delta_list)
        hyperweight['last_dec_output'] = new_decoder_param_list
        for i, (prefix, new_decoder_param) in enumerate(zip(decoders_prefix, new_decoder_param_list)):
            if i >= n_st:
                tmp = i - n_st; k = 0
                while tmp >= n_mt_tasks[k]: tmp -= n_mt_tasks[k]; k += 1
                model_idx = n_st + k
            else: model_idx = i
            for layer in new_decoder_param.keys():
                update_ckpt[model_idx][prefix + '.' + layer] = new_decoder_param[layer]
        del last_decoder_param_list, decoder_delta_list
    else:
        raise NotImplementedError

    for model_idx in range(N):
        keys_to_del = [k for k in update_ckpt[model_idx].keys() if 'dklm_feat' in k]
        for k in keys_to_del: del update_ckpt[model_idx][k]

    update_ckpt = move_ckpt(update_ckpt, 'cuda')
    for model_idx in range(N):
        all_clients[model_idx]['model'].module.load_state_dict(update_ckpt[model_idx], strict=False)

    del update_ckpt

def update_hyperweight(all_nets, hyperweight, save_ckpt, last_ckpt):
    '''
    Update hyperweights with corresponding delta of encoder and decoder parameters
    '''
    if 'enc' in hyperweight.keys():
        encoder_param_list, encoder_keys, enc_layers, enc_shapes = get_encoder_params(all_nets, save_ckpt)
        last_encoder_param_list, _, _, _ = get_encoder_params(all_nets, last_ckpt)
        diff_list = get_delta_dict_list(last_encoder_param_list, encoder_param_list)
        flatten_diff = flatten_param(diff_list, enc_layers)
        hyperweight['enc'].train()
        optimizer = hyperweight['enc_optimizer']
        optimizer.zero_grad()
        torch.autograd.backward(hyperweight['last_enc_output'], flatten_diff, retain_graph=True)
        optimizer.step()

    if 'dec' in hyperweight.keys():
        decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = get_decoder_params(all_nets, save_ckpt)
        last_decoder_param_list, last_decoders_prefix, _, _, _ = get_decoder_params(all_nets, last_ckpt)
        assert decoders_prefix == last_decoders_prefix
        diff_list = get_delta_dict_list(last_decoder_param_list, decoder_param_list)
        hyperweight['dec'].train()
        optimizer = hyperweight['dec_optimizer']
        optimizer.zero_grad()
        for i in range(len(decoder_param_list)):
            last_output = list(map(lambda x: hyperweight['last_dec_output'][i][x], dec_layers))
            diff_param = list(map(lambda x: diff_list[i][x], dec_layers))
            torch.autograd.backward(last_output, diff_param, retain_graph=True)
        optimizer.step()