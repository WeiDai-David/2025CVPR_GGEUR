import os
import numpy as np
from tqdm import tqdm

# 加载合并的类索引
def load_combined_class_indices(class_indices_file):
    """
    加载已合并的类索引文件
    """
    return np.load(class_indices_file, allow_pickle=True).item()

# 加载客户端索引
def load_client_indices(client_indices_path):
    """
    加载某个客户端的索引文件
    """
    return np.load(client_indices_path)

# 匹配客户端索引和类索引
def match_client_class_indices(client_indices, class_indices):
    """
    将客户端的索引与类索引匹配，生成客户端在每个类的索引
    """
    client_class_indices = {class_label: [] for class_label in class_indices.keys()}

    for index in client_indices:
        # 查找该索引属于哪个类
        for class_label, class_idx_list in class_indices.items():
            if index in class_idx_list:
                client_class_indices[class_label].append(index)
                break

    return client_class_indices

# 保存客户端每个类的索引
def save_client_class_indices(client_class_indices, output_dir, client_id):
    """
    保存每个客户端在每个类中的索引到 .npy 和 .txt 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    for class_label, indices in client_class_indices.items():
        # 保存为 npy 文件
        np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_indices.npy'), indices)
        # 保存为 txt 文件
        with open(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_indices.txt'), 'w') as f:
            f.write(f"Client {client_id}, Class {class_label} indices: {list(indices)}\n")

# 主函数
def main():
    base_dir = './output_indices'  # 保存的索引文件目录
    datasets = ['mnist', 'usps', 'svhn', 'syn']  # 数据集名称
    output_base_dir = './output_client_class_indices'  # 保存客户端每个类的索引
    os.makedirs(output_base_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"正在处理 {dataset_name} 数据集...")

        # 加载合并的类索引文件
        combined_class_indices_file = os.path.join(base_dir, f'{dataset_name}/combined_class_indices.npy')
        class_indices = load_combined_class_indices(combined_class_indices_file)

        # 遍历每个客户端
        client_indices_dir = os.path.join(base_dir, f'{dataset_name}')
        for client_file in tqdm(os.listdir(client_indices_dir), desc=f"处理 {dataset_name} 客户端"):
            if client_file.startswith('client_') and client_file.endswith('_indices.npy'):
                client_id = client_file.split('_')[1]  # 提取客户端编号
                client_indices_path = os.path.join(client_indices_dir, client_file)

                # 加载客户端分配到的索引
                client_indices = load_client_indices(client_indices_path)

                # 将客户端的索引进一步分配到各个类
                client_class_indices = match_client_class_indices(client_indices, class_indices)

                # 保存每个客户端每个类的索引
                output_dir = os.path.join(output_base_dir, dataset_name)
                save_client_class_indices(client_class_indices, output_dir, client_id)

if __name__ == '__main__':
    main()
