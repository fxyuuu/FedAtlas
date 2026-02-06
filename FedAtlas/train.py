import argparse
import copy
import datetime
import os
import shutil
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from aggregate import aggregate, update_hyperweight
from datasets.custom_dataset import get_dataloader, get_dataset
from losses import get_criterion
from models.build_models import build_model
from models.hyperweight import HyperAggWeight, HyperCrossAttention

# [FedAtlas] 引入 DKLM 和 SemCom 模块
from models.dklm import HyperDKLM
from models.semcom import FedAtlasSemCom # [新增]

from train_utils import eval_metric, local_train
from utils import (
    RunningMeter,
    create_results_dir,
    get_loss_metric,
    get_mt_config,
    get_st_config,
    move_ckpt,
    set_seed,
)

# 导入自定义工具模块
from partition_utils import get_dominant_labels, dirichlet_partition
from comm_utils import get_model_size, estimate_comm_time, apply_loss_to_state_dict, get_parameter_dimensions

# 获取模型维度信息的辅助函数
def get_parameter_dimensions(model):
    """
    返回: (总参数数量, 详细的维度字典字符串)
    """
    dims = {}
    total_params = 0
    for name, param in model.state_dict().items():
        shape = list(param.size())
        dims[name] = shape
        total_params += param.numel()
    return total_params, str(dims)

def main(args, all_clients, hyperweight=None, local_rank=0, semcom_module=None):
    N = len(all_clients)
    # Setup loss meters
    train_loss = {}
    val_loss = {}
    for idx in range(N):
        train_loss[idx] = {}
        val_loss[idx] = {}
        for task in all_clients[idx]['tasks']:
            train_loss[idx][task] = RunningMeter()
            val_loss[idx][task] = RunningMeter()

    # Save last_ckpt
    last_ckpt = []
    for idx in range(N):
        # 自适应获取 state_dict
        model = all_clients[idx]['model']
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        last_ckpt.append(copy.deepcopy(state_dict))
    if args.save_vram:
        last_ckpt = move_ckpt(last_ckpt, 'cpu')
    save_ckpt = copy.deepcopy(last_ckpt)

    # Create hyperweight log
    if local_rank == 0:
        if args.encoder_agg == "conflict_averse":
            if args.save_vram:
                enc_hw = hyperweight['enc']
            else:
                enc_hw = hyperweight['enc'].module
            alpha = enc_hw.alpha.detach().cpu().numpy().tolist()
            with open(os.path.join(args.exp_dir, 'enc_alpha.txt'), 'w') as f:
                f.write(str(alpha) + '\n')

        if args.decoder_agg == "cross_attention":
            if args.save_vram:
                dec_hw = hyperweight['dec']
            else:
                dec_hw = hyperweight['dec'].module
            beta = dec_hw.beta
            beta_list = [beta[key].detach().cpu().numpy().tolist() for key in dec_hw.beta_names]
            with open(os.path.join(args.exp_dir, 'dec_beta.txt'), 'w') as f:
                f.write(str(dec_hw.beta_names) + '\n')
                f.write(str(beta_list) + '\n')

    # =========================================================================
    # [System Metrics] 初始化累加器和日志文件
    # =========================================================================
    cumulative_comm_volume = 0.0   # 总流量累加
    cumulative_upload_volume = 0.0 

    if local_rank == 0:
        print(f"--> [System Metrics Mode] Bandwidth={args.bandwidth}MB/s")
        
        # 1. 创建 CSV 文件
        metrics_csv_path = os.path.join(args.exp_dir, 'system_metrics.csv')
        mode = 'a' if args.resume_path else 'w'
        
        if not args.resume_path:
            with open(metrics_csv_path, mode) as f:
                header = [
                    "Round",
                    "Client_Upload_MB",
                    "Client_Total_MB",
                    "Round_All_Upload_MB",
                    "Round_All_Total_MB",
                    "Acc_Upload_MB",
                    "Acc_Total_MB",
                    "Param_Count",
                    "Train_Time(s)",
                    "Agg_Time(s)",
                    "Round_Total_Time(s)"
                ]
                f.write(",".join(header) + "\n")
        
        # 2. 创建文本文件记录维度
        dims_log_path = os.path.join(args.exp_dir, 'model_dimensions.txt')
        if not args.resume_path:
            with open(dims_log_path, 'w') as f:
                f.write("=== Model Parameter Dimensions Log ===\n")
            
    # 循环从 start_round 开始
    for cr in range(args.start_round, args.max_rounds):
        start_time = time.time()
        t_round_start = time.time()
        logs = {}
        t_train_start = time.time()
        
        for idx in range(N):
            # [修改] 明确定义 DKLM 开关
            is_dklm_active = (args.encoder_agg == 'dklm') or (args.decoder_agg == 'dklm')

            # Train clients' local models
            local_train(idx=idx,
                        cr=cr,
                        train_loss=train_loss[idx],
                        local_rank=local_rank,
                        fp16=args.fp16,
                        mu=args.mu,
                        use_dklm=is_dklm_active,
                        encoder_agg=args.encoder_agg,
                        # [新增] 传入语义通信模块
                        semcom_module=semcom_module,
                        **all_clients[idx])

            train_stats = get_loss_metric(train_loss[idx], all_clients[idx]['tasks'], 'train', idx)
            logs.update(train_stats)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_train_end = time.time()
        train_duration = t_train_end - t_train_start
        
        # =========================================================================
        # 模拟上传过程中的数据丢失
        # =========================================================================
        for idx in range(N):
            model = all_clients[idx]['model']
            if hasattr(model, 'module'):
                clean_state_dict = model.module.state_dict()
            else:
                clean_state_dict = model.state_dict()
            
            if args.loss_rate > 0:
                device = all_clients[idx]['model'].device_ids[0] if hasattr(all_clients[idx]['model'], 'device_ids') else 'cuda'
                dirty_state_dict = apply_loss_to_state_dict(clean_state_dict, args.loss_rate, device)
                save_ckpt[idx] = dirty_state_dict
            else:
                save_ckpt[idx] = copy.deepcopy(clean_state_dict)
                
        if args.save_vram:
            save_ckpt = move_ckpt(save_ckpt, 'cpu')

        # Update hyperweight
        if cr > 0:
            update_hyperweight(all_clients, hyperweight, save_ckpt, last_ckpt)
            if local_rank == 0:
                if args.encoder_agg == "conflict_averse":
                    if args.save_vram: enc_hw = hyperweight['enc']
                    else: enc_hw = hyperweight['enc'].module
                    alpha = enc_hw.alpha.detach().cpu().numpy().tolist()
                    with open(os.path.join(args.exp_dir, 'enc_alpha.txt'), 'a') as f:
                        f.write(str(alpha) + '\n')

                if args.decoder_agg == "cross_attention":
                    if args.save_vram: dec_hw = hyperweight['dec']
                    else: dec_hw = hyperweight['dec'].module
                    beta = dec_hw.beta
                    beta_list = [beta[key].detach().cpu().numpy().tolist() for key in dec_hw.beta_names]
                    with open(os.path.join(args.exp_dir, 'dec_beta.txt'), 'a') as f:
                        f.write(str(beta_list) + '\n')

        # =====================================================================
        # [计时点 C] 服务端聚合
        # =====================================================================
        t_agg_start = time.time()
        
        aggregate(
            all_clients,
            save_ckpt,
            last_ckpt,
            hyperweight,
            args.encoder_agg,
            args.decoder_agg,
            args.ca_c,
            save_dir=args.results_dir,
            current_round=cr,
            local_rank=local_rank
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_agg_end = time.time()
        agg_duration = t_agg_end - t_agg_start

        # Update last_ckpt
        for idx in range(N):
            last_ckpt[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
        if args.save_vram:
            last_ckpt = move_ckpt(last_ckpt, 'cpu')

        # [计时点 A结束] 轮次结束
        t_round_end = time.time()
        round_total_duration = t_round_end - t_round_start
        
        if local_rank == 0:
            model_size_mb = get_model_size(all_clients[0]['model'])
            client_upload = model_size_mb 
            # 如果开启 SemCom，流量会大幅减少
            if args.semcom_enabled:
                # 粗略估算：压缩比 = sparsity_ratio * (quant_bits / 32)
                ratio = args.semcom_sparsity * (args.semcom_quant_bits / 32.0)
                client_upload = client_upload * ratio
            
            client_total_vol = client_upload + model_size_mb # 下行通常发完整模型
            
            round_all_upload = client_upload * N
            round_all_total_vol = client_total_vol * N
            
            cumulative_upload_volume += round_all_upload
            cumulative_comm_volume += round_all_total_vol
            
            param_count, shape_str = get_parameter_dimensions(all_clients[0]['model'])
            
            print(f"--> [Round {cr} Metrics] Time: Train={train_duration:.2f}s, Agg={agg_duration:.2f}s")
            print(f"    Vol: Upload={round_all_upload:.2f}MB, AccUpload={cumulative_upload_volume:.2f}MB")
            
            with open(metrics_csv_path, 'a') as f:
                line = [
                    str(cr),
                    f"{client_upload:.4f}",       
                    f"{client_total_vol:.4f}",    
                    f"{round_all_upload:.4f}",    
                    f"{round_all_total_vol:.4f}", 
                    f"{cumulative_upload_volume:.4f}", 
                    f"{cumulative_comm_volume:.4f}",   
                    str(param_count),
                    f"{train_duration:.4f}",
                    f"{agg_duration:.4f}",
                    f"{round_total_duration:.4f}"
                ]
                f.write(",".join(line) + "\n")
            
            with open(dims_log_path, 'w') as f:
                f.write(f"=== Round {cr} Model Dimensions ===\n")
                f.write(f"Total Parameters: {param_count}\n")
                f.write("Detailed Shapes:\n")
                f.write(shape_str + "\n")

            print("CR %d finishs, Time: %.1fs." % (cr, time.time() - start_time))
                                                                                 
            if (cr + 1) == args.max_rounds or (cr + 1) % args.eval_freq == 0:
                print('Validation at CR %d.' % cr)
                val_logs = {}
                for idx in range(N):
                    res = eval_metric(idx=idx, **all_clients[idx])
                    val_logs.update(res)
                print(val_logs)
                
                val_history_file = os.path.join(args.exp_dir, 'val_history.txt')
                with open(val_history_file, 'a') as f:
                    f.write(f"=== Round {cr} ===\n")
                    for k, v in val_logs.items():
                        f.write(f"{k}: {v}\n")
                    f.write("\n")
                    
                if args.wandb_name is not None:
                    wandb.log({**logs, **val_logs})

                save_ckpt_temp = {}
                for idx in range(N):
                    save_ckpt_temp[idx] = copy.deepcopy(all_clients[idx]['model'].module.state_dict())
                torch.save(save_ckpt_temp, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
                print('Checkpoint saved.')
                del save_ckpt_temp
            else:
                if args.wandb_name is not None:
                    wandb.log(logs)

    if local_rank == 0:
        print('Training finished.')


def get_clients(args, model_config, client_configs, local_rank):
    """
    Get clients from configs
    """
    all_clients = []
    n_decoders = 0

    for dataname in client_configs:
        client_config = client_configs[dataname]
        net_task_dataidx_map, n_clients = (
            client_config['net_task_dataidx_map'],
            client_config['n_clients'],
        )
        # ================== Dirichlet Partition ==================
        if args.alpha is not None:
            if local_rank == 0:
                print(f"Applying Dirichlet partition with alpha={args.alpha} for {dataname}")
            
            full_ds = get_dataset(
                dataname=dataname,
                train=True,
                tasks=['semseg'], 
                transform=None,
                dataidxs=None,
                local_rank=local_rank
            )
            labels = get_dominant_labels(full_ds, dataname)
            new_dataidxs_map = dirichlet_partition(labels, n_clients, args.alpha)

            for i in range(n_clients):
                net_task_dataidx_map[i]['dataidx'] = new_dataidxs_map[i]
        # ==============================================================

        for idx in range(n_clients):
            task_list = net_task_dataidx_map[idx]['task_list']
            dataidxs = net_task_dataidx_map[idx]['dataidx']

            train_ds_local = get_dataset(
                dataname=dataname,
                train=True,
                tasks=task_list,
                transform=client_config['train_transforms'],
                dataidxs=dataidxs,
                local_rank=local_rank,
            )
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds_local, drop_last=True)
            train_dl_local = get_dataloader(
                train=True,
                configs=client_config,
                dataset=train_ds_local,
                sampler=train_sampler,
            )

            val_ds_local = get_dataset(
                dataname=dataname,
                train=False,
                tasks=task_list,
                transform=client_config['val_transforms'],
                local_rank=local_rank,
            )
            val_dl_local = get_dataloader(train=False, configs=client_config, dataset=val_ds_local)

            model = build_model(task_list, dataname, **model_config).cuda()
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
            
            if client_config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=float(client_config['lr']),
                    momentum=0.9,
                    weight_decay=float(client_config['weight_decay']),
                )
            elif client_config['optimizer'] == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(client_config['lr']),
                    weight_decay=float(client_config['weight_decay']),
                )
            else:
                raise NotImplementedError("Invalid optimizer %s!" % client_config['optimizer'])

            max_epochs = int(args.max_rounds) * int(client_config['local_epochs'])
            warmup_epochs = int(client_config['warmup_epochs'])
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=max_epochs - warmup_epochs,
                lr_min=1.25e-6,
                warmup_t=warmup_epochs,
                warmup_lr_init=1.25e-7,
                warmup_prefix=True,
            )
            client = {}
            client['tasks'] = task_list
            client['dataname'] = dataname
            client['train_dl'] = train_dl_local
            client['val_dl'] = val_dl_local
            client['local_epochs'] = client_config['local_epochs']
            client['model'] = model
            client['optimizer'] = optimizer
            client['scheduler'] = scheduler
            client['criterion'] = get_criterion(dataname, task_list).cuda()
            client['scaler'] = torch.cuda.amp.GradScaler(enabled=args.fp16)

            all_clients.append(client)
            n_decoders += len(task_list)

    return all_clients, n_decoders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Config file path")
    parser.add_argument('--exp', type=str, required=True, help="Experiment name")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, help="Wandb project name")
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16')
    parser.add_argument('--save_vram', action='store_true', help='Whether to save vram')

    parser.add_argument('--max_rounds', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=5)

    parser.add_argument('--encoder_agg', default='conflict_averse', help="none,fedavg")
    parser.add_argument('--ca_c', type=float, default=0.4)
    parser.add_argument('--enc_alpha_init', type=float, default=0.1)
    parser.add_argument('--decoder_agg', default='cross_attention', help="none,fedavg")
    parser.add_argument('--dec_beta_init', type=float, default=0.1)
                          
    parser.add_argument('--alpha', type=float, default=None, help='Dirichlet distribution alpha')
    parser.add_argument('--bandwidth', type=float, default=10.0, help="Bandwidth (MB/s)")
    parser.add_argument('--latency', type=float, default=0.02, help="Network latency")
    parser.add_argument('--loss_rate', type=float, default=0.0, help="Packet loss rate (0.0-1.0)")

    parser.add_argument('--dklm_k', type=int, default=8, help="DKLM clusters")
    parser.add_argument('--dklm_alpha', type=float, default=2.0, help="DKLM manifold weight")
    parser.add_argument('--dklm_beta', type=float, default=1.0, help="DKLM relaxation")
    parser.add_argument('--dklm_gamma', type=float, default=0.1, help="DKLM reg weight")
    parser.add_argument('--mu', type=float, default=0.0, help='FedProx mu parameter')

    # ================= [新增] 断点续训参数 =================
    parser.add_argument('--resume_path', type=str, default=None, help="Path to checkpoint.pth to resume from")
    parser.add_argument('--start_round', type=int, default=0, help="Round to start from (e.g., 96)")
    # =======================================================

    # ================= [新增] FedAtlas 语义通信参数 =================
    parser.add_argument('--semcom_enabled', action='store_true', help="Enable Semantic Communication")
    parser.add_argument('--semcom_sparsity', type=float, default=0.1, help="Top-S sparsity ratio (e.g. 0.1)")
    parser.add_argument('--semcom_quant_bits', type=int, default=8, help="Quantization bits (e.g. 8)")
    parser.add_argument('--semcom_dsr', action='store_true', help="Enable DSR for feature compression")
    parser.add_argument('--semcom_codebook_size', type=int, default=64, help="VQ Codebook size")
    # ==============================================================
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        exp_config = yaml.safe_load(stream)

    exp_config = {**exp_config, **vars(args)}

    set_seed(args.seed)
    dist.init_process_group('nccl', timeout=datetime.timedelta(0, 3600 * 2))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    if local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if args.resume_path is None:
            args.exp_dir, args.checkpoint_dir = create_results_dir(args.results_dir, args.exp)
            shutil.copy(args.config_path, os.path.join(args.exp_dir, 'config.yml'))
        else:
            args.exp_dir = os.path.join(args.results_dir, args.exp)
            args.checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
            print(f"Resuming experiment in: {args.exp_dir}")

        if args.wandb_name is not None:
            import wandb
            wandb.init(project=args.wandb_name, id=args.exp, name=args.exp, config=exp_config, resume="allow")
    dist.barrier()

    client_configs = {}
    if 'ST_Datasets' in exp_config:
        client_configs.update(get_st_config(exp_config['ST_Datasets'], local_rank))
    if 'MT_Datasets' in exp_config:
        client_configs.update(get_mt_config(exp_config['MT_Datasets'], local_rank))

    all_clients, n_decoders = get_clients(args, exp_config['Model'], client_configs, local_rank)

    # 续训逻辑
    if args.resume_path is not None:
        if local_rank == 0:
            print(f"🔥🔥🔥 Loading checkpoint from {args.resume_path}...")
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        for idx in range(len(all_clients)):
            all_clients[idx]['model'].load_state_dict(checkpoint[idx], strict=False)
        if local_rank == 0:
            print(f"✅ Checkpoint loaded! Starting from round {args.start_round}")

    # [新增] 初始化 SemCom 模块
    semcom_module = None
    if args.semcom_enabled:
        if local_rank == 0:
            print(f"🚀 Initializing FedAtlas SemCom (Sparsity={args.semcom_sparsity}, Quant={args.semcom_quant_bits}bit)")
        semcom_module = FedAtlasSemCom(device=torch.device(f'cuda:{local_rank}'), args=args)

    hyperweight = {}
    if args.encoder_agg == 'dklm' or args.decoder_agg == 'dklm':
        if local_rank == 0:
            print(f"Initializing DKLM (FedAtlas) with k={args.dklm_k}")
        dklm_net = HyperDKLM(num_clusters=args.dklm_k, alpha=args.dklm_alpha, beta=args.dklm_beta, gamma=args.dklm_gamma, device='cuda')
        hyperweight['dklm_module'] = dklm_net
    if args.encoder_agg == "conflict_averse":
        hypernet = HyperAggWeight(K=len(all_clients), init_alpha=args.enc_alpha_init)
        if args.save_vram: hyperweight['enc'] = hypernet
        else: hyperweight['enc'] = DDP(hypernet.cuda(), device_ids=[local_rank])
        hyperweight['enc_optimizer'] = torch.optim.SGD(hypernet.parameters(), **exp_config['Hyperweight'])

    if args.decoder_agg == "cross_attention":
        dummy_decoder = all_clients[0]['model'].module.decoders
        hypernet = HyperCrossAttention(model=dummy_decoder, K=n_decoders, init_beta=args.dec_beta_init)
        if args.save_vram: hyperweight['dec'] = hypernet
        else: hyperweight['dec'] = DDP(hypernet.cuda(), device_ids=[local_rank])
        hyperweight['dec_optimizer'] = torch.optim.SGD(hypernet.parameters(), **exp_config['Hyperweight'])

    main(
        args=args,
        all_clients=all_clients,
        hyperweight=hyperweight,
        local_rank=local_rank,
        semcom_module=semcom_module, # [新增]
    )
    dist.destroy_process_group()