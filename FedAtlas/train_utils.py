import torch
from tqdm import tqdm
import torch.nn.functional as F

from evaluation.evaluate_utils import PerformanceMeter
from utils import get_output, to_cuda

import copy
import time

# [新增] 引入语义通信模块
from models.semcom import FedAtlasSemCom

# -----------------------------------------------------------------------------
# DKLM 专用函数 (FedAvg 运行时不会调用此函数)
# -----------------------------------------------------------------------------
def extract_gradient_features(model, dataloader, criterion, device, num_batches=10):
    # 初始化容器，防止 NameError
    final_features = {}
    feature_accumulator = {}
    count_accumulator = {}
    handle = None

    model.eval()
    model.zero_grad()
    
    # 1. 自动定位 backbone 或 encoder
    target_module = None
    if hasattr(model, 'module'): mod = model.module
    else: mod = model
    for name in ['backbone', 'encoder', 'body', 'features']:
        if hasattr(mod, name):
            target_module = getattr(mod, name)
            break
    if target_module is None:
        try: target_module = list(mod.children())[0]
        except: return {} 

    # 2. Hook
    features_storage = {}
    def hook_fn(m, i, o):
        if isinstance(o, (list, tuple)): feat = o[-1]
        else: feat = o
        if not feat.requires_grad: feat.requires_grad_(True)
        feat.retain_grad()
        features_storage['feat'] = feat
    handle = target_module.register_forward_hook(hook_fn)

    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            batch = to_cuda(batch)
            images = batch['image']
            
            with torch.set_grad_enabled(True):
                outputs = model(images)
                if hasattr(criterion, 'tasks'): task_list = criterion.tasks
                else: task_list = outputs.keys()
                available_tasks = [t for t in task_list if t in batch]
                
                for task in available_tasks:
                    preds = outputs[task]
                    gt = batch[task]
                    # 尺寸对齐
                    if gt.dim() >= 3:
                        target_h, target_w = gt.shape[-2], gt.shape[-1]
                        if preds.shape[-2] != target_h or preds.shape[-1] != target_w:
                            is_3d = (preds.dim() == 3)
                            if is_3d: preds = preds.unsqueeze(1)
                            preds = F.interpolate(preds, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            if is_3d: preds = preds.squeeze(1)

                    loss = None
                    if hasattr(criterion, 'loss_ft') and task in criterion.loss_ft:
                         loss = criterion.loss_ft[task](preds, gt)
                    if loss is None: continue

                    if 'feat' in features_storage and features_storage['feat'].grad is not None:
                        features_storage['feat'].grad.zero_()
                    
                    loss.backward(retain_graph=True)
                    
                    if 'feat' in features_storage:
                        grad = features_storage['feat'].grad
                        if grad is not None:
                            # GAP
                            if grad.dim() == 4: g = grad.mean(dim=[0, 2, 3])
                            elif grad.dim() == 3: g = grad.mean(dim=[0, 1])
                            else: g = grad.mean(dim=0)
                            g = g.detach().cpu()
                            
                            # [关键修改 1] 温和的数值清洗 (只处理 NaN，不截断正常值)
                            g = torch.nan_to_num(g, nan=0.0)

                            if task not in feature_accumulator:
                                feature_accumulator[task] = g; count_accumulator[task] = 1
                            else:
                                feature_accumulator[task] += g; count_accumulator[task] += 1
                            
    except Exception as e:
        print(f"[Warning] DKLM grad extraction error: {e}")
    finally:
        if handle is not None: handle.remove()
        model.zero_grad()
    
    # [关键修改 2] 严格的 L2 归一化
    # 保证所有特征向量长度为 1，这样后面的点积就等于余弦相似度，绝对不会爆炸
    for t, v in feature_accumulator.items():
        if count_accumulator[t] > 0:
            avg = v / count_accumulator[t]
            final_features[f"dklm_feat_{t}"] = F.normalize(avg, p=2, dim=0)

    return final_features
    
def local_train(idx, cr, local_epochs, tasks, train_dl, model, optimizer, scheduler, criterion, scaler, train_loss,
                local_rank, fp16, mu=0.0, semcom_module=None, **args): # [修改] 增加 semcom_module
    """
    Train local_epochs on the client model
    """
    use_dklm = args.get('use_dklm', False)
    
    # =========================================================
    # [新增] PSR 准备阶段: 记录初始权重
    # =========================================================
    initial_state_dict = {}
    if semcom_module is not None:
        # 只记录需要更新的参数 (requires_grad=True)
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_state_dict[name] = param.detach().clone()

    model.train()

    # [修改2] FedProx: 如果 mu > 0，训练前先把全局模型备份一份，作为约束锚点
    global_model = None
    if mu > 0:
        # 深拷贝当前模型作为全局模型参考
        # 注意：如果 model 是 DDP 包装的，deepcopy 也会复制 wrapper，结构是一致的
        global_model = copy.deepcopy(model)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

    for epoch in range(local_epochs):
        train_dl.sampler.set_epoch(cr * local_epochs + epoch)
        for batch in tqdm(train_dl,
                          desc="CR %d Local Epoch %d Net %d Task: %s" % (cr, epoch, idx, ",".join(tasks)),
                          disable=(local_rank != 0)):
            optimizer.zero_grad()
            batch = to_cuda(batch)
            images = batch['image']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                outputs = model(images)
                loss_dict = criterion(outputs, batch, tasks)
                # [修改3] FedProx 核心逻辑: 计算近端项 (Proximal Term)
                if mu > 0 and global_model is not None:
                    proximal_term = 0.0
                    # 遍历当前可学习参数(w)和全局参考参数(w_t)
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2) # L2 范数
                    
                    # 将正则项加入总 Loss: Loss = Task_Loss + (mu / 2) * ||w - w_t||^2
                    loss_dict['total'] += (mu / 2.0) * proximal_term

            # Log loss values
            for task in tasks:
                loss_value = loss_dict[task].detach().item()
                batch_size = outputs[task].size(0)
                train_loss[task].update(loss_value / batch_size, batch_size)

            scaler.scale(loss_dict['total']).backward()
            # [新增] 1. 先解包梯度 (Unscale)
            scaler.unscale_(optimizer)

            # [新增] 2. 强力梯度裁剪 (防止梯度爆炸导致特征提取出 Inf)
            # max_norm 建议设为 1.0 或 5.0。如果坍塌严重，设为 1.0。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # [修改] 3. 更新参数
            # scaler.step(optimizer)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step(cr * local_epochs + epoch)

    # =========================================================
    # [新增] PSR 应用阶段: 语义压缩参数更新
    # =========================================================
    if semcom_module is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_state_dict:
                    # 1. 计算原始更新量 Delta = W_new - W_old
                    raw_update = param.data - initial_state_dict[name]
                    
                    # 2. 应用语义压缩 (Error Feedback + TopS + Quant)
                    compressed_update = semcom_module.compress_model_update(idx, name, raw_update)
                    
                    # 3. 将"有损"的更新量加回初始权重，模拟 Server 接收到的状态
                    # W_uploaded = W_old + Delta_compressed
                    param.data = initial_state_dict[name] + compressed_update
        
        # 释放内存
        del initial_state_dict

    # =========================================================
    # DKLM 逻辑开关 (集成 DSR)
    # =========================================================
    # 只有当显式开启 DKLM 时才运行，FedAvg 模式下直接跳过
    if use_dklm:
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        try:
            # 1. 提取原始特征
            grad_feats = extract_gradient_features(model, train_dl, criterion, device, num_batches=5)
            
            if hasattr(model, 'module'):
                mod = model.module
            else:
                mod = model
                
            for k, v in grad_feats.items():
                v = v.to(device)
                
                # =================================================
                # [新增] DSR 应用阶段: 特征压缩
                # =================================================
                if semcom_module is not None and semcom_module.use_dsr:
                    # 获取 task name, key 格式通常是 "dklm_feat_semseg"
                    task_name = k.replace("dklm_feat_", "")
                    # 进行 VQ 压缩
                    v_compressed = semcom_module.compress_task_feature(idx, task_name, v)
                    # 替换原始特征
                    v = v_compressed

                # 注册到 Buffer
                if hasattr(mod, k):
                    getattr(mod, k).copy_(v)
                else:
                    mod.register_buffer(k, v)
                    
        except Exception as e:
            # 捕获异常，确保即使 DKLM 失败，训练也不会崩溃
            print(f"[Warning] DKLM feature extraction failed: {e}")
            pass # 继续流程

def eval_metric(tasks, dataname, val_dl, model, idx, **args):
    """
    Evaluate client model
    """

    performance_meter = PerformanceMeter(dataname, tasks)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating Net %d Task: %s" % (idx, ",".join(tasks))):
            batch = to_cuda(batch)
            images = batch['image']
            outputs = model.module(images)
            performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

    eval_results = performance_meter.get_score()

    results_dict = {}
    for task in tasks:
        for key in eval_results[task]:
            results_dict['eval/' + str(idx) + '_' + task + '_' + key] = eval_results[task][key]

    return results_dict