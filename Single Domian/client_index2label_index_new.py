# client_index2label_index.py 该脚本用于将best_client_guidance.py生成的各个类指导的客户端集合转换成各个客户端的索引
# 具体的通过各个客户端的分配数据索引文件和CIFAR-10的类索引文件对齐
# 注意这里是客户端集合，但是转换的时候仍然是各个客户端的索引，这是因为虽然同属同一类的指导数据，但是在不同客户端之间要进行聚合协方差
import numpy as np
import os
import re


# 解析.npy
def load_client_indices(file_path):
    return np.load(file_path)


# 从文件中加载类索引
def load_class_indices(file_path):
    # 初始化一个包含10个空列表的列表
    all_class_indices = [[] for _ in range(10)]

    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式匹配每个类的索引
    for class_idx in range(10):
        pattern = re.compile(rf'Class {class_idx} indices: \[(.*?)\]', re.DOTALL)
        match = pattern.search(content)
        if match:
            indices_str = match.group(1).replace('\n', '').replace('...', '')
            indices = list(map(int, indices_str.split()))
            all_class_indices[class_idx] = indices

    return all_class_indices


# 根据对数据集data_batch进行类索引划分后的索引，将其作为指导从类下客户端中挑选
def get_class_indices(client_indices, class_indices):
    return [idx for idx in client_indices if idx in class_indices]


def load_selected_clients(file_path):
    selected_clients = {}
    with open(file_path, 'r') as file:
        for line in file:
            if 'selected client' in line:
                parts = line.split(':')
                try:
                    class_idx = int(parts[0].split(' ')[3])  # 获取类索引
                except ValueError as e:
                    print(f"Error parsing line: {line}")
                    print(f"parts[0].split(' '): {parts[0].split(' ')}")
                    raise e
                clients = list(map(int, parts[1].strip().strip('[]').split(',')))
                selected_clients[class_idx] = clients
    return selected_clients


def convert_to_dataset_indices(selected_clients, dataset_name, alpha):
    dataset_indices = {class_idx: {} for class_idx in range(10)}
    class_indices_path = f'./{dataset_name}/context/alpha={alpha}_class_indices.txt'
    all_class_indices = load_class_indices(class_indices_path)

    for class_idx, clients in selected_clients.items():
        class_indices = all_class_indices[class_idx]

        for client in clients:
            file_path = f'./{dataset_name}/alpha={alpha}_{dataset_name}_client_{client}_indices.npy'
            client_indices = load_client_indices(file_path)
            class_client_indices = get_class_indices(client_indices, class_indices)
            if client not in dataset_indices[class_idx]:
                dataset_indices[class_idx][client] = []
            dataset_indices[class_idx][client].extend(class_client_indices)

    return dataset_indices


def main():
    dataset_name = 'CIFAR-10'  # 可更改为 'CIFAR-100' 或 'TinyImageNet'
    alpha = 0.1  # Dirichlet 分布参数
    # 根据数据放置的文件夹结构
    selected_clients_file_path = f'./{dataset_name}/context/alpha={alpha}_selected_clients_for_each_class.txt'

    # 调试信息，确保文件路径正确
    print(f"Selected clients file path: {selected_clients_file_path}")

    selected_clients_for_each_class = load_selected_clients(selected_clients_file_path)

    # 调试信息，确保选定的客户端正确加载
    print(f"Selected clients for each class: {selected_clients_for_each_class}")

    dataset_indices = convert_to_dataset_indices(selected_clients_for_each_class, dataset_name, alpha)

    # 保存和打印结果
    for class_idx, clients_data in dataset_indices.items():
        for client_idx, indices in clients_data.items():
            print(f'Class {class_idx} client {client_idx} number of samples: {len(indices)}')
            # 输出样本数 此时同一类所有输出的客户端的样本数应该满足0.8的关系
            np.savetxt(
                f'./{dataset_name}/context/best_guidance_alpha={alpha}_class_{class_idx}_client_{client_idx}_indices.txt',
                indices, fmt='%d')
            np.save(f'./{dataset_name}/best_guidance_alpha={alpha}_class_{class_idx}_client_{client_idx}_indices.npy',
                    indices)
            # 调试信息，确保数据正确
            print(f'Class {class_idx} client {client_idx} indices: {indices}')


if __name__ == "__main__":
    main()
