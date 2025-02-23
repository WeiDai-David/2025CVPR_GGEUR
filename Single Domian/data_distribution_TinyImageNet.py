# import os
# import numpy as np
# import pickle
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from matplotlib.colors import LinearSegmentedColormap
# from torchvision.datasets import ImageFolder
# import json

# class TinyImageNet_truncated(Dataset):
#     def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
#         self.root = root
#         self.dataidxs = dataidxs
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.dataset = ImageFolder(root=self.root, transform=self.transform)
#         self.data, self.targets = zip(*[(data, target) for data, target in self.dataset])
#         self.data = list(self.data)
#         self.targets = list(self.targets)
#         self.class_indices = self._get_class_indices()
#         if self.dataidxs is not None:
#             self.data = [self.data[idx] for idx in self.dataidxs]
#             self.targets = [self.targets[idx] for idx in self.dataidxs]

#     def _get_class_indices(self):
#         class_indices = {}
#         for idx, (_, target) in enumerate(self.dataset):
#             if target not in class_indices:
#                 class_indices[target] = []
#             class_indices[target].append(idx)
#         return class_indices

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         return img, target

#     def __len__(self):
#         return len(self.data)

# # 加载 TinyImageNet 数据
# def load_tinyimagenet_data(datadir):
#     transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
#     train_dataset = TinyImageNet_truncated(root=f'{datadir}/train', train=True, transform=transform)
#     val_dataset = TinyImageNet_truncated(root=f'{datadir}/new_val', train=False, transform=transform)
#     X_train, y_train, class_indices = train_dataset.data, train_dataset.targets, train_dataset.class_indices
#     X_val, y_val = val_dataset.data, val_dataset.targets
#     return (X_train, y_train, class_indices, X_val, y_val)

# # 使用 Dirichlet 分布划分数据，并确保每个客户端的数据量满足最小要求
# def partition_data(labels, num_clients=10, alpha=0.5, min_require_size=1):
#     n = len(labels)
#     class_indices = [np.where(np.array(labels) == i)[0] for i in range(200)]
#     client_indices = [[] for _ in range(num_clients)]

#     # 先分配每个类的至少 min_require_size 个样本给每个客户端
#     for k in range(200):
#         idx_k = class_indices[k]
#         np.random.shuffle(idx_k)
#         for client in range(num_clients):
#             client_indices[client].extend(idx_k[client * min_require_size:(client + 1) * min_require_size])
#         class_indices[k] = idx_k[num_clients * min_require_size:]  # 剩下的样本

#     # 使用 Dirichlet 分布分配剩余的样本
#     for k in range(200):
#         idx_k = class_indices[k]
#         if len(idx_k) > 0:
#             proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
#             proportions = np.round(proportions / proportions.sum() * len(idx_k)).astype(int)
#             proportions[-1] = len(idx_k) - sum(proportions[:-1])
#             split_indices = np.split(idx_k, np.cumsum(proportions[:-1]))
#             for i in range(num_clients):
#                 client_indices[i].extend(split_indices[i].tolist())

#     return client_indices

# # 可视化客户端的标签分布
# def plot_label_distribution(clients_indices, dataset, save_path=None):
#     labels = [label for _, label in dataset]
#     labels = np.array(labels)
#     num_clients = 10
#     distributions = np.zeros((200, num_clients))
#     for i, indices in enumerate(clients_indices):
#         client_labels = labels[indices]
#         label_counts = np.bincount(client_labels, minlength=200)
#         distributions[:, i] = label_counts
    
#     # 自定义颜色映射
#     colors = [(1, 1, 1), (1, 0, 0)]  # 从白色到红色
#     n_bins = 100  # 使用100个颜色段
#     cmap_name = 'white_to_red'
#     cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

#     fig, ax = plt.subplots(figsize=(15, 12))
#     sns.heatmap(distributions, annot=False, ax=ax, cmap=cm, fmt='g')
#     plt.title("Label distribution across clients")
#     plt.xlabel("Client Index")
#     plt.ylabel("Class Label")
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

# def save_class_mapping(datadir, output_dir):
#     wnid_map_file = os.path.join(datadir, 'wnids.txt')
#     class_map_file = os.path.join(output_dir, 'class_map.txt')

#     # 读取WNID文件
#     with open(wnid_map_file, 'r') as f:
#         wnids = [line.strip() for line in f.readlines()]

#     # 生成0-199到WNID的映射
#     class_map = {i: wnid for i, wnid in enumerate(wnids)}

#     # 保存映射到文件
#     with open(class_map_file, 'w') as f:
#         for idx, wnid in class_map.items():
#             f.write(f"{idx}\t{wnid}\n")

#     print(f"Class mapping saved to {class_map_file}")

# # 主函数
# def main():
#     datadir = './data/tiny-imagenet-200'
#     num_clients = 10
#     alpha = 0.3  # Dirichlet 分布参数
#     X_train, y_train, class_indices, X_val, y_val = load_tinyimagenet_data(datadir)
#     clients_indices = partition_data(y_train, num_clients, alpha)

#     # 创建输出目录
#     output_dir = "./TinyImageNet/context"
#     os.makedirs(output_dir, exist_ok=True)

#     # 计算并保存每个类在所有客户端中的总数量
#     combined_client_labels = np.concatenate([np.array(y_train)[client_indices] for client_indices in clients_indices])
#     total_class_counts = np.bincount(combined_client_labels, minlength=200)

#     # 保存和打印每个类的总数
#     with open(os.path.join(output_dir, f"alpha={alpha}_class_counts.txt"), "w") as f:
#         for i, count in enumerate(total_class_counts):
#             print(f"Class {i}: Total {count} items")
#             f.write(f"Class {i}: Total {count} items\n")

#     # 保存和打印每个客户端的类分布
#     with open(os.path.join(output_dir, f"alpha={alpha}_client_class_distribution.txt"), "w") as f:
#         for i, client_indices in enumerate(clients_indices):
#             client_label_counts = np.bincount(np.array(y_train)[client_indices], minlength=200)
#             print(f"Client {i} class distribution: {client_label_counts}")
#             f.write(f"Client {i} class distribution: {client_label_counts}\n")

#     # 保存和打印每个客户端的索引
#     with open(os.path.join(output_dir, f"alpha={alpha}_client_indices.txt"), "w") as f:
#         for i, indices in enumerate(clients_indices):
#             np.save(os.path.join(f'./TinyImageNet/alpha={alpha}_TinyImageNet_client_{i}_indices.npy'), indices)  # 将索引保存到文件
#             print(f'Client {i} indices: {indices}')
#             f.write(f'Client {i} indices: {indices}\n')

#     # 保存和打印每个类的索引
#     np.set_printoptions(threshold=np.inf)  # 设置打印选项为显示所有值
#     with open(os.path.join(output_dir, f"alpha={alpha}_class_indices.txt"), "w") as f:
#         for class_label, indices in class_indices.items():
#             np.save(os.path.join(f'./TinyImageNet/class_{class_label}_indices.npy'), indices)  # 将索引保存到文件
#             f.write(f'Class {class_label} indices: {indices}\n')
#             print(f'Class {class_label} indices: {indices}')
#     np.set_printoptions(threshold=1000)  # 恢复默认打印选项

#     # 绘制标签分布图并保存
#     plot_label_distribution(clients_indices, [(x, y) for x, y in zip(X_train, y_train)],
#                             save_path=f'./TinyImageNet/images/alpha={alpha}_label_distribution_heatmap.png')

#     # 保存类编号到TinyImageNet编号的映射
#     save_class_mapping(datadir, output_dir)

# if __name__ == "__main__":
#     main()
    
    
import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from torchvision.datasets import ImageFolder
import json

class TinyImageNet_truncated(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = ImageFolder(root=self.root, transform=self.transform)
        self.data, self.targets = zip(*[(data, target) for data, target in self.dataset])
        self.data = list(self.data)
        self.targets = list(self.targets)
        self.class_indices = self._get_class_indices()
        if self.dataidxs is not None:
            self.data = [self.data[idx] for idx in self.dataidxs]
            self.targets = [self.targets[idx] for idx in self.dataidxs]

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, target) in enumerate(self.dataset):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)
        return class_indices

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

# 加载 TinyImageNet 数据
def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = TinyImageNet_truncated(root=f'{datadir}/train', train=True, transform=transform)
    val_dataset = TinyImageNet_truncated(root=f'{datadir}/new_val', train=False, transform=transform)
    X_train, y_train, class_indices = train_dataset.data, train_dataset.targets, train_dataset.class_indices
    X_val, y_val = val_dataset.data, val_dataset.targets
    return (X_train, y_train, class_indices, X_val, y_val)

# 使用 Dirichlet 分布划分数据，并确保每个客户端的数据量满足最小要求
def partition_data(labels, num_clients=10, alpha=0.5, min_require_size=1):
    n = len(labels)
    class_indices = [np.where(np.array(labels) == i)[0] for i in range(200)]
    client_indices = [[] for _ in range(num_clients)]

    # 先分配每个类的至少 min_require_size 个样本给每个客户端
    for k in range(200):
        idx_k = class_indices[k]
        np.random.shuffle(idx_k)
        for client in range(num_clients):
            client_indices[client].extend(idx_k[client * min_require_size:(client + 1) * min_require_size])
        class_indices[k] = idx_k[num_clients * min_require_size:]  # 剩下的样本

    # 使用 Dirichlet 分布分配剩余的样本
    for k in range(200):
        idx_k = class_indices[k]
        if len(idx_k) > 0:
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.round(proportions / proportions.sum() * len(idx_k)).astype(int)
            proportions[-1] = len(idx_k) - sum(proportions[:-1])
            split_indices = np.split(idx_k, np.cumsum(proportions[:-1]))
            for i in range(num_clients):
                client_indices[i].extend(split_indices[i].tolist())

    return client_indices

# 可视化客户端的标签分布
def plot_label_distribution(clients_indices, dataset, save_path=None):
    labels = [label for _, label in dataset]
    labels = np.array(labels)
    num_clients = 10
    distributions = np.zeros((200, num_clients))
    for i, indices in enumerate(clients_indices):
        client_labels = labels[indices]
        label_counts = np.bincount(client_labels, minlength=200)
        distributions[:, i] = label_counts
    
    # 自定义颜色映射
    colors = [(1, 1, 1), (1, 0, 0)]  # 从白色到红色
    n_bins = 100  # 使用100个颜色段
    cmap_name = 'white_to_red'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(distributions, annot=False, ax=ax, cmap=cm, fmt='g')
    plt.title("Label distribution across clients")
    plt.xlabel("Client Index")
    plt.ylabel("Class Label")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 生成类映射
def generate_class_mapping(new_val_dir):
    class_dirs = [d for d in os.listdir(new_val_dir) if os.path.isdir(os.path.join(new_val_dir, d))]
    class_dirs_sorted = sorted(class_dirs, key=lambda x: int(x[1:]))  # 去除 'n' 后按数值排序
    class_map = {i: class_dirs_sorted[i] for i in range(len(class_dirs_sorted))}
    return class_map

# 保存类映射到文件
def save_class_mapping(class_map, output_dir):
    class_map_file = os.path.join(output_dir, 'class_map.txt')
    with open(class_map_file, 'w') as f:
        for idx, wnid in class_map.items():
            f.write(f"{idx}\t{wnid}\n")
    print(f"Class mapping saved to {class_map_file}")

# 主函数
def main():
    datadir = './data/tiny-imagenet-200'
    num_clients = 10
    alpha = 0.09  # Dirichlet 分布参数
    X_train, y_train, class_indices, X_val, y_val = load_tinyimagenet_data(datadir)
    clients_indices = partition_data(y_train, num_clients, alpha)

    # 创建输出目录
    output_dir = "./TinyImageNet/context"
    os.makedirs(output_dir, exist_ok=True)

    # 计算并保存每个类在所有客户端中的总数量
    combined_client_labels = np.concatenate([np.array(y_train)[client_indices] for client_indices in clients_indices])
    total_class_counts = np.bincount(combined_client_labels, minlength=200)

    # 保存和打印每个类的总数
    with open(os.path.join(output_dir, f"alpha={alpha}_class_counts.txt"), "w") as f:
        for i, count in enumerate(total_class_counts):
            print(f"Class {i}: Total {count} items")
            f.write(f"Class {i}: Total {count} items\n")

    # 保存和打印每个客户端的类分布
    with open(os.path.join(output_dir, f"alpha={alpha}_client_class_distribution.txt"), "w") as f:
        for i, client_indices in enumerate(clients_indices):
            client_label_counts = np.bincount(np.array(y_train)[client_indices], minlength=200)
            print(f"Client {i} class distribution: {client_label_counts}")
            f.write(f"Client {i} class distribution: {client_label_counts}\n")

    # 保存和打印每个客户端的索引
    with open(os.path.join(output_dir, f"alpha={alpha}_client_indices.txt"), "w") as f:
        for i, indices in enumerate(clients_indices):
            np.save(os.path.join(f'./TinyImageNet/alpha={alpha}_TinyImageNet_client_{i}_indices.npy'), indices)  # 将索引保存到文件
            print(f'Client {i} indices: {indices}')
            f.write(f'Client {i} indices: {indices}\n')

    # 保存和打印每个类的索引
    np.set_printoptions(threshold=np.inf)  # 设置打印选项为显示所有值
    with open(os.path.join(output_dir, f"alpha={alpha}_class_indices.txt"), "w") as f:
        for class_label, indices in class_indices.items():
            np.save(os.path.join(f'./TinyImageNet/class_{class_label}_indices.npy'), indices)  # 将索引保存到文件
            f.write(f'Class {class_label} indices: {indices}\n')
            print(f'Class {class_label} indices: {indices}')
    np.set_printoptions(threshold=1000)  # 恢复默认打印选项

    # 绘制标签分布图并保存
    plot_label_distribution(clients_indices, [(x, y) for x, y in zip(X_train, y_train)],
                            save_path=f'./TinyImageNet/images/alpha={alpha}_label_distribution_heatmap.png')

    # 生成类映射
    class_map = generate_class_mapping(os.path.join(datadir, 'new_val'))

    # 保存类编号到TinyImageNet编号的映射
    save_class_mapping(class_map, output_dir)

if __name__ == "__main__":
    main()
