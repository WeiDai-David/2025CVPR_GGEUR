import numpy as np
import os
import argparse

def process_client_class_indices(dataset, alpha):
    # 定义文件路径
    base_dir = f'./{dataset}'
    if dataset == 'TinyImageNet':
        num_clients = 10
        num_classes = 200
    else:
        num_clients = 10
        num_classes = 10 if dataset == 'CIFAR-10' else 100

    client_indices_files = [os.path.join(base_dir, f'alpha={alpha}_{dataset}_client_{i}_indices.npy') for i in range(num_clients)]
    class_indices_files = [os.path.join(base_dir, f'class_{i}_indices.npy') for i in range(num_classes)]

    # 创建存储结果的目录
    output_dir = os.path.join(base_dir, 'client_class_indices')
    os.makedirs(output_dir, exist_ok=True)

    # 交叉引用客户端和类索引
    for client_id, client_file in enumerate(client_indices_files):
        # 加载客户端索引
        client_indices = np.load(client_file)

        for class_id, class_file in enumerate(class_indices_files):
            # 加载类索引
            class_indices = np.load(class_file)

            # 获取交集
            client_class_indices = np.intersect1d(client_indices, class_indices)

            # 保存结果
            output_file = os.path.join(output_dir, f'alpha={alpha}_{dataset}_client_{client_id}_class_{class_id}_indices.npy')
            np.save(output_file, client_class_indices)
            print(f'Saved {output_file}, total indices: {len(client_class_indices)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TinyImageNet', choices=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'], help='The dataset to process.')
    parser.add_argument('--alpha', type=float, default=0.09, help='The alpha value for Dirichlet distribution.')
    args = parser.parse_args()

    process_client_class_indices(args.dataset, args.alpha)
