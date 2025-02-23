import os
import argparse
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np

# 设置随机种子以确保可重复性
torch.manual_seed(0)

# 定义数据集加载函数
def load_dataset(dataset_name, data_dir='./data'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'USPS':
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.USPS(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    elif dataset_name == 'SYN':
        # SYN数据集需要自行下载和预处理，此处提供一个占位符示例
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # 假设SYN数据集已存在于 data_dir/SYN 目录下，且为 torchvision.datasets.ImageFolder 格式
        train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'SYN'), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset

# 生成索引和标签文件
def generate_index_files(dataset_name, dataset):
    indices_per_class = {i: [] for i in range(10)}

    for idx, (image, label) in enumerate(dataset):
        indices_per_class[int(label)].append(idx)

    output_dir = f'./{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    for class_idx, indices in indices_per_class.items():
        np.save(os.path.join(output_dir, f'class_{class_idx}_indices.npy'), indices)
        np.save(os.path.join(output_dir, f'class_{class_idx}_labels.npy'), [class_idx] * len(indices))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate index files for selected dataset')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'USPS', 'SVHN', 'SYN'], default='USPS', help='Dataset to use (default: MNIST)')
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset = load_dataset(dataset_name)
    generate_index_files(dataset_name, dataset)
    print(f'Index files for {dataset_name} dataset have been generated.')
