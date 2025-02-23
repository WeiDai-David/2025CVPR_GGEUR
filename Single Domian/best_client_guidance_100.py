# 针对cifar-100，由于有100个类，所以一个客户端生成的txt将不会只占据一行，而是多行，因此检索的时候需要变换
# best_client_guidance.py 本脚本用于根据data_distribution_CIFAR-10.py生成的各个客户端的类分布,生成各个类指导的客户端集合
import numpy as np
import os
import argparse

# 加载各个客户端类数量分布：client_class_distribution.txt文件
def load_client_data(file_path):
    client_data = []
    with open(file_path, 'r') as file:
        client_distribution = []
        for line in file:
            if line.startswith('Client'):
                if client_distribution:
                    client_data.append(client_distribution)
                    client_distribution = []
                data = line.split(':')[1].strip().strip('[]').split()
                data = list(map(int, data))
                client_distribution.extend(data)
            else:
                data = line.strip().strip('[]').split()
                data = list(map(int, data))
                client_distribution.extend(data)
        if client_distribution:
            client_data.append(client_distribution)
    return client_data

# 遍历所有类，分析类下所有客户端，按照降序排列客户端编号，获取累加样本数量大于等于threshold比例的客户端列表
def analyze_class_distribution(client_data, num_classes, total_class_items, threshold=0.8):
    all_selected_clients = {}
    for class_idx in range(num_classes):
        class_data = [(i, client[class_idx]) for i, client in enumerate(client_data)]
        class_data.sort(key=lambda x: x[1], reverse=True)
        cumulative_sum = 0
        selected_clients = []
        for client_id, count in class_data:
            cumulative_sum += count
            selected_clients.append(client_id)
            if cumulative_sum >= total_class_items * threshold:
                break
        all_selected_clients[class_idx] = selected_clients
        print(f"Class {class_idx} total items: {cumulative_sum}")
    return all_selected_clients

def main(dataset_type, alpha):
    datasets_info = {
        "CIFAR-10": {
            "base_path": './CIFAR-10/context/',
            "num_classes": 10,
            "total_class_items": 5000,
        },
        "CIFAR-100": {
            "base_path": './CIFAR-100/context/',
            "num_classes": 100,
            "total_class_items": 500,
        },
        "TinyImageNet": {
            "base_path": './TinyImageNet/context/',
            "num_classes": 200,
            "total_class_items": 500,
        }
    }

    if dataset_type not in datasets_info:
        print(f"Unsupported dataset type: {dataset_type}")
        return

    info = datasets_info[dataset_type]
    file_path = f"{info['base_path']}alpha={alpha}_client_class_distribution.txt"
    num_classes = info["num_classes"]
    total_class_items = info["total_class_items"]

    # 确保目录存在
    os.makedirs(info["base_path"], exist_ok=True)

    client_data = load_client_data(file_path)
    selected_clients_for_each_class = analyze_class_distribution(client_data, num_classes, total_class_items)

    output_file = f"{info['base_path']}alpha={alpha}_selected_clients_for_each_class.txt"
    with open(output_file, "w") as f:
        for class_idx, selected_clients in selected_clients_for_each_class.items():
            print(f"{dataset_type}-Class_{class_idx}_selected_clients: {selected_clients}")
            f.write(f"{dataset_type}-Class_{class_idx}_selected_clients: {selected_clients}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TinyImageNet', choices=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'],
                        help='The type of dataset to analyze.')
    parser.add_argument('--alpha', type=float, default=0.2, help='The alpha value for Dirichlet distribution.')
    args = parser.parse_args()
    main(args.dataset, args.alpha)
