import argparse
import os
import cv2  # [必须] 用于形态学操作提取边缘
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from evaluation.evaluate_utils import PerformanceMeter, predict
from models.build_models import build_model
from utils import create_pred_dir, get_mt_config, get_output, get_st_config, to_cuda

# -----------------------------------------------------------------------------
# [增强版] Edge ODSF 计算函数 (带自动修复逻辑)
# -----------------------------------------------------------------------------
def compute_odsf_edge(predictions, targets, thresholds=50):
    print(f">>> [Debug] Computing ODSF on {len(predictions)} samples with Tolerance...")
    
    thresh_vals = np.linspace(0.01, 0.99, thresholds)
    total_tp = np.zeros(len(thresh_vals))
    total_fp = np.zeros(len(thresh_vals))
    total_fn = np.zeros(len(thresh_vals))

    # 定义容差核 (3x3)，相当于允许 1 像素的误差
    tolerance_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for i, p in enumerate(tqdm(predictions, desc="Processing Images")):
        t = targets[i].squeeze()
        p = p.squeeze()

        # 1. 自动转换 GT (语义块 -> 边缘线)
        # ----------------------------------------------------------
        # 如果是实心掩码 (非零像素 > 5%)，强制转边缘
        if np.mean(t > 0) > 0.05: 
            if t.max() <= 1: t_uint8 = (t * 255).astype(np.uint8)
            else: t_uint8 = t.astype(np.uint8)
            
            # 使用 Canny 算子可能更细，但 morphological gradient 更稳健
            # 这里我们用 morphological gradient
            dilation = cv2.dilate(t_uint8, tolerance_kernel, iterations=1)
            erosion = cv2.erode(t_uint8, tolerance_kernel, iterations=1)
            edge = cv2.absdiff(dilation, erosion)
            t = (edge > 0).astype(np.bool_)
        else:
            # 已经是边缘了，二值化
            t = (t > 0.1).astype(np.bool_)
        # ----------------------------------------------------------

        # 2. [核心修改] 创建“容差 GT” (Dilated GT)
        # ----------------------------------------------------------
        # 将细线 GT 膨胀 1 次，变成约 3 像素宽的带子
        # 只要预测落在这个带子里，就算 TP (True Positive)
        t_uint8 = t.astype(np.uint8)
        t_dilated = cv2.dilate(t_uint8, tolerance_kernel, iterations=1).astype(np.bool_)
        
        # 同时，为了公平计算 Precision (防止预测太粗占便宜)，
        # 我们也可以把预测图里的"骨架"提取出来比，或者允许 Prediction 也有容差。
        # 但最简单的标准做法是：Match = (Pred & Dilated_GT)
        # ----------------------------------------------------------

        for k, thresh in enumerate(thresh_vals):
            p_binary = (p > thresh).astype(np.bool_)
            
            # [关键修改] 计算指标
            # TP: 预测为1，且落在 GT 的容差范围内 (Match)
            # 注意：这里用 t_dilated 来判定“是否命中”
            match = p_binary & t_dilated
            tp = np.count_nonzero(match)
            
            # FP: 预测为1，但完全没落在 GT 附近
            # FP = Total_Pred - TP
            fp = np.count_nonzero(p_binary) - tp
            
            # FN: GT 为1，但周围没有预测 (Miss)
            # 为了计算 FN，我们要反过来：看有多少 GT 像素没被 Pred 覆盖
            # 这里可以简单近似：FN = Total_GT - (Pred_Dilated & GT)
            # 或者简单地：FN = Total_GT - (Prediction & GT_Dilated) 的重叠数
            # 最严格的定义是：FN = count(GT) - TP_matched_to_GT
            # 简单写法：
            fn = np.count_nonzero(t) - tp
            
            # 修正 FN 不能小于 0 (因为 dilation 可能导致 tp > count(t))
            if fn < 0: fn = 0

            total_tp[k] += tp
            total_fp[k] += fp
            total_fn[k] += fn

    best_f1 = 0.0
    for k in range(len(thresh_vals)):
        tp = total_tp[k]
        fp = total_fp[k]
        fn = total_fn[k]
        if tp == 0: f1 = 0.0
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1: best_f1 = f1

    print(f">>> [Result] Best ODSF (with tolerance): {best_f1:.4f}")
    return best_f1

# -----------------------------------------------------------------------------
# 修改后的 eval_metric (增加尺寸对齐)
# -----------------------------------------------------------------------------
def eval_metric(idx, dataname, tasks, test_dl, model, evaluate, save, pred_dir, **args):
    if evaluate:
        performance_meter = PerformanceMeter(dataname, tasks)

    if save:
        tasks_to_save = tasks
    else:
        tasks_to_save = ['edge'] if 'edge' in tasks else []

    assert evaluate or len(tasks_to_save) > 0

    # 开关
    store_edge_odsf = evaluate and ('edge' in tasks)
    edge_preds_buffer = []
    edge_targs_buffer = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating Net %d Task: %s" % (idx, ",".join(tasks))):
            batch = to_cuda(batch)
            images = batch['image']
            outputs = model(images)

            if evaluate:
                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

                # [修改] 收集 Edge 数据，增加尺寸对齐
                if store_edge_odsf and 'edge' in outputs:
                    pred = outputs['edge'] # [B, 1, H, W]
                    target = batch['edge'] # [B, H, W]
                    
                    # 1. 强制尺寸对齐 (以 Target 为准)
                    if pred.shape[-2:] != target.shape[-2:]:
                        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    
                    pred = torch.sigmoid(pred)
                    if pred.dim() == 4: pred = pred.squeeze(1)
                    
                    # 2. 存入 buffer (转 CPU)
                    batch_p = pred.cpu().numpy()
                    batch_t = target.cpu().numpy()
                    
                    # 逐个样本 append，适应不同 batch size
                    for p_img, t_img in zip(batch_p, batch_t):
                        edge_preds_buffer.append(p_img)
                        edge_targs_buffer.append(t_img)

            for task in tasks_to_save:
                predict(dataname, batch['meta'], outputs, task, pred_dir, idx)

    if evaluate:
        eval_results = performance_meter.get_score()

        results_dict = {}
        for t in tasks:
            for key in eval_results[t]:
                results_dict[str(idx) + '_' + t + '_' + key] = eval_results[t][key]
        
        # [新增] 计算 ODSF
        if store_edge_odsf and len(edge_preds_buffer) > 0:
            odsf_score = compute_odsf_edge(edge_preds_buffer, edge_targs_buffer)
            results_dict[str(idx) + '_edge_odsf'] = odsf_score
            print(f"Client {idx} Edge ODSF: {odsf_score:.4f}")

        return results_dict


def test(args, all_clients):
    '''
    Test all clients with test data
    '''
    test_results = {}
    for idx in range(len(all_clients)):
        res = eval_metric(idx=idx, evaluate=args.evaluate, save=args.save, pred_dir=args.pred_dir, **all_clients[idx])
        if args.evaluate:
            test_results.update({key: "%.5f" % res[key] for key in res})

    # log results
    if args.evaluate:
        print(test_results)
        results_file = os.path.join(args.results_dir, args.exp, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write(str(test_results))


def get_clients(client_configs, model_config):
    all_clients = []
    for dataname in client_configs:
        client_config = client_configs[dataname]
        net_task_dataidx_map, n_clients = (
            client_config['net_task_dataidx_map'],
            client_config['n_clients'],
        )

        for idx in range(n_clients):
            task_list = net_task_dataidx_map[idx]['task_list']

            test_ds_local = get_dataset(
                dataname=dataname,
                train=False,
                tasks=task_list,
                transform=client_config['val_transforms'],
            )
            test_dl_local = get_dataloader(train=False, configs=client_config, dataset=test_ds_local)
            
            # Build model
            model = build_model(task_list, dataname, **model_config).cuda()
            
            client = {}
            client['tasks'] = task_list
            client['dataname'] = dataname
            client['test_dl'] = test_dl_local
            client['model'] = model

            all_clients.append(client)
    return all_clients


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='experiment name')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('--evaluate', action='store_true', help='Whether to evaluate all clients')
    parser.add_argument('--save', action='store_true', help='Whether to save predictions')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')

    args = parser.parse_args()

    with open(os.path.join(args.results_dir, args.exp, 'config.yml'), 'r') as stream:
        exp_config = yaml.safe_load(stream)

    torch.cuda.set_device(args.gpu_id)

    client_configs = {}
    if 'ST_Datasets' in exp_config:
        client_configs.update(get_st_config(exp_config['ST_Datasets']))
    if 'MT_Datasets' in exp_config:
        client_configs.update(get_mt_config(exp_config['MT_Datasets']))

    all_clients = get_clients(client_configs, exp_config['Model'])
    args.checkpoint_dir, args.pred_dir = create_pred_dir(args.results_dir, args.exp, all_clients)

    checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_file):
        raise ValueError('Checkpoint %s not found!' % (checkpoint_file))

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    for idx in range(len(all_clients)):
        all_clients[idx]['model'].load_state_dict(checkpoint[idx], strict=False)

    test(args, all_clients)