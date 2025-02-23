import numpy as np
import os
import re
import argparse

# 定义特征提取函数
def calculate_mean(Z):
    """
    计算矩阵Z的均值
    """
    return np.mean(Z, axis=0)

def calculate_covariance(Z, mean_Z):
    """
    计算矩阵Z的协方差
    """
    Z_centered = Z - mean_Z
    return (1 / Z.shape[0]) * np.dot(Z_centered.T, Z_centered)

def process_multiple_appearances(class_idx, sub_dirs, class_feature_data):
    """
    处理同一类在多个客户端中的情况
    """
    print("process_multiple_appearances")
    sample_counts = []
    mean_matrices = []
    covariance_matrices = []

    for sub_dir in sub_dirs:
        if f'class_{class_idx}' in sub_dir:
            file_path = os.path.join(sub_dir, 'final_embeddings.npy')
            Z = np.load(file_path)
            mean_Z = calculate_mean(Z)
            cov_matrix = calculate_covariance(Z, mean_Z)
            sample_counts.append(Z.shape[0])
            mean_matrices.append(mean_Z)
            covariance_matrices.append(cov_matrix)

    total_samples = np.sum(sample_counts)
    combined_mean = np.sum([sample_counts[i] * mean_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    combined_cov_matrix = np.sum([sample_counts[i] * covariance_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    combined_cov_matrix += np.sum([sample_counts[i] * np.outer(mean_matrices[i] - combined_mean, mean_matrices[i] - combined_mean) for i in range(len(sample_counts))], axis=0) / total_samples

    class_feature_data[class_idx] = combined_cov_matrix

def process_single_appearance(class_idx, sub_dirs, class_feature_data):
    """
    处理同一类仅在一个客户端中的情况
    """
    print("process_single_appearance")
    for sub_dir in sub_dirs:
        if f'class_{class_idx}' in sub_dir:
            file_path = os.path.join(sub_dir, 'final_embeddings.npy')
            Z = np.load(file_path)
            mean_Z = calculate_mean(Z)
            cov_matrix = calculate_covariance(Z, mean_Z)
            class_feature_data[class_idx] = cov_matrix

def main(dataset_choice, alpha):
    """
    主函数：处理所有索引文件并提取图像特征
    """
    base_dir = f'./{dataset_choice}/features/resnet_alpha={alpha}_guide/'
    os.makedirs(base_dir, exist_ok=True)
    output_cov_dir = f'./{dataset_choice}/features/resnet_alpha={alpha}_cov_matrix'
    num_classes = 10 if dataset_choice.lower() == 'cifar-10' else 100 if dataset_choice.lower() == 'cifar-100' else 200

    sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    class_counts = {}
    for sub_dir in sub_dirs:
        match = re.search(r'class_(\d+)', sub_dir)
        if match:
            class_idx = int(match.group(1))
            if class_idx in class_counts:
                class_counts[class_idx] += 1
            else:
                class_counts[class_idx] = 1

    class_feature_data = [np.zeros((512, 512)) for _ in range(num_classes)]  # 初始化为零矩阵
    for class_idx, count in class_counts.items():
        if count >= 2:
            process_multiple_appearances(class_idx, sub_dirs, class_feature_data)
        else:
            process_single_appearance(class_idx, sub_dirs, class_feature_data)

    os.makedirs(output_cov_dir, exist_ok=True)
    for idx, feature_data in enumerate(class_feature_data):
        print(f"Class {idx} 数据特征:")
        np.set_printoptions(threshold=np.inf)
        # print(f"协方差矩阵:\n{feature_data}\n")
        np.set_printoptions(threshold=1000)

        txt_path = f'{output_cov_dir}/class_{idx}_cov_matrix.txt'
        with open(txt_path, 'w') as f:
            f.write(str(feature_data.tolist()))

        npy_path = f'{output_cov_dir}/class_{idx}_cov_matrix.npy'
        np.save(npy_path, feature_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'], help='The dataset to process.')
    parser.add_argument('--alpha', type=float, default=5, help='The alpha value for Dirichlet distribution.')
    args = parser.parse_args()

    main(args.dataset, args.alpha)
