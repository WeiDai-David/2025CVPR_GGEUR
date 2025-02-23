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

# 定义 CIFAR100_truncated 类
class CIFAR100_truncated(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target, self.class_indices = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = datasets.CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        class_indices = [np.where(target == i)[0] for i in range(100)]
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            class_indices = [np.where(target == i)[0] for i in range(100)]
        return data, target, class_indices

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

# 加载 CIFAR-100 数据
def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)
    X_train, y_train, class_indices = cifar100_train_ds.data, cifar100_train_ds.target, cifar100_train_ds.class_indices
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target
    return (X_train, y_train, class_indices, X_test, y_test)

# 使用 Dirichlet 分布划分数据，并确保每个客户端的数据量满足最小要求
def partition_data(labels, num_clients=10, alpha=0.5, min_require_size=1):
    n = len(labels)
    class_indices = [np.where(labels == i)[0] for i in range(100)]
    client_indices = [[] for _ in range(num_clients)]

    # 先分配每个类的至少 min_require_size 个样本给每个客户端
    for k in range(100):
        idx_k = class_indices[k]
        np.random.shuffle(idx_k)
        for client in range(num_clients):
            client_indices[client].extend(idx_k[client * min_require_size:(client + 1) * min_require_size])
        class_indices[k] = idx_k[num_clients * min_require_size:]  # 剩下的样本

    # 使用 Dirichlet 分布分配剩余的样本
    for k in range(100):
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
    distributions = np.zeros((100, num_clients))
    for i, indices in enumerate(clients_indices):
        client_labels = labels[indices]
        label_counts = np.bincount(client_labels, minlength=100)
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

# 主函数
def main():
    datadir = './data'
    num_clients = 10
    alpha = 50  # Dirichlet 分布参数
    X_train, y_train, class_indices, X_test, y_test = load_cifar100_data(datadir)
    clients_indices = partition_data(y_train, num_clients, alpha)

    # 创建输出目录
    output_dir = "./CIFAR-100/context"
    os.makedirs(output_dir, exist_ok=True)

    # 计算并保存每个类在所有客户端中的总数量
    combined_client_labels = np.concatenate([y_train[client_indices] for client_indices in clients_indices])
    total_class_counts = np.bincount(combined_client_labels, minlength=100)

    # 保存和打印每个类的总数
    with open(os.path.join(output_dir, f"alpha={alpha}_class_counts.txt"), "w") as f:
        for i, count in enumerate(total_class_counts):
            print(f"Class {i}: Total {count} items")
            f.write(f"Class {i}: Total {count} items\n")

    # 保存和打印每个客户端的类分布
    with open(os.path.join(output_dir, f"alpha={alpha}_client_class_distribution.txt"), "w") as f:
        for i, client_indices in enumerate(clients_indices):
            client_label_counts = np.bincount(y_train[client_indices], minlength=100)
            print(f"Client {i} class distribution: {client_label_counts}")
            f.write(f"Client {i} class distribution: {client_label_counts}\n")

    # 保存和打印每个客户端的索引
    with open(os.path.join(output_dir, f"alpha={alpha}_client_indices.txt"), "w") as f:
        for i, indices in enumerate(clients_indices):
            np.save(os.path.join(f'./CIFAR-100/alpha={alpha}_CIFAR-100_client_{i}_indices.npy'), indices)  # 将索引保存到文件
            print(f'Client {i} indices: {indices}')
            f.write(f'Client {i} indices: {indices}\n')

    # 保存和打印每个类的索引
    np.set_printoptions(threshold=np.inf)  # 设置打印选项为显示所有值
    with open(os.path.join(output_dir, f"alpha={alpha}_class_indices.txt"), "w") as f:
        for i, indices in enumerate(class_indices):
            np.save(os.path.join(f'./CIFAR-100/class_{i}_indices.npy'), indices)  # 将索引保存到文件
            print(f'Class {i} indices: {indices}')
            f.write(f'Class {i} indices: {indices}\n')
    np.set_printoptions(threshold=1000)  # 恢复默认打印选项

    # 绘制标签分布图并保存
    plot_label_distribution(clients_indices, [(x, y) for x, y in zip(X_train, y_train)],
                            save_path=f'./CIFAR-100/images/alpha={alpha}_label_distribution_heatmap.png')

if __name__ == "__main__":
    main()
