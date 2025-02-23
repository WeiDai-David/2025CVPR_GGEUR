import os
import numpy as np
import shutil
import argparse

def load_selected_clients(file_path, dataset):
    selected_clients = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(f'{dataset}-Class_'):
                parts = line.split(':')
                class_info = parts[0].strip().replace(f'{dataset}-Class_', '').replace('_selected_clients', '')
                class_idx = int(class_info.strip())
                client_info = parts[1].strip().strip('[]').split(',')
                client_indices = [int(client.strip()) for client in client_info]
                selected_clients[class_idx] = client_indices
    return selected_clients

def copy_selected_features(selected_clients, base_dir, output_dir, alpha, dataset):
    for class_idx, client_indices in selected_clients.items():
        for client_idx in client_indices:
            # 将路径修改为 resnet_initial
            input_dir = os.path.join(base_dir, f'resnet_initial/alpha={alpha}_class_{class_idx}_client_{client_idx}')
            output_class_dir = os.path.join(output_dir, f'alpha={alpha}_class_{class_idx}_client_{client_idx}')
            os.makedirs(output_class_dir, exist_ok=True)

            # 复制特征文件和标签文件
            for file_name in ['final_embeddings.npy', 'labels.npy']:
                src_file = os.path.join(input_dir, file_name)
                dst_file = os.path.join(output_class_dir, file_name)
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
                    print(f'Copied {src_file} to {dst_file}')
                else:
                    print(f'{src_file} does not exist.')

def main(dataset, alpha):
    base_dir = f'./{dataset}/features'
    txt_file_path = f'./{dataset}/context/alpha={alpha}_selected_clients_for_each_class.txt'
    output_dir = f'./{dataset}/features/resnet_alpha={alpha}_guide'

    selected_clients = load_selected_clients(txt_file_path, dataset)
    copy_selected_features(selected_clients, base_dir, output_dir, alpha, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'], help='The dataset to process.')
    parser.add_argument('--alpha', type=float, default=5, help='The alpha value for Dirichlet distribution.')
    args = parser.parse_args()

    main(args.dataset, args.alpha)
