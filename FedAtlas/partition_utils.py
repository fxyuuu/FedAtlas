# partition_utils.py
import numpy as np

def get_dominant_labels(dataset, dataname):
    """
    提取数据集的主导标签，用于 Dirichlet 分布划分
    """
    labels = []
    print(f"Extracting labels for {dataname} partition...")
    
    ignore_index = 255
    for i in range(len(dataset)):
        # 尝试直接调用内部方法加载 mask，避免不必要的 transform
        if hasattr(dataset, '_load_semseg'):
            mask = dataset._load_semseg(i)
        else:
            sample = dataset[i]
            if 'semseg' in sample:
                mask = sample['semseg'].numpy()
            else:
                labels.append(0)
                continue

        mask = mask.flatten()
        mask = mask[mask != ignore_index]
        
        if len(mask) == 0:
            labels.append(0)
        else:
            counts = np.bincount(mask.astype(int))
            labels.append(np.argmax(counts))
            
    return np.array(labels)


def dirichlet_partition(labels, n_clients, alpha, min_require_size=10):
    """
    基于 Dirichlet 分布的数据划分算法
    """
    n_classes = len(np.unique(labels))
    n_data = len(labels)
    client_id_map = {}
    min_size = 0

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            
            # 平衡处理
            proportions = np.array([p * (len(idx_j) < n_data / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            
        min_size = min([len(idx_j) for idx_j in idx_batch])

    for i in range(n_clients):
        client_id_map[i] = idx_batch[i]
        
    return client_id_map