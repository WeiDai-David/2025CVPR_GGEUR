# clip_image_tensor2aggregate_covariance_matrix.py 该脚本用于处理单一指导和协同指导,将特征转换成协方差矩阵和聚合协方差矩阵
import numpy as np
import os
import re

# 定义路径
alpha = 0.5
base_dir = './cifar-10/features/alpha={alpha}_guide/'

# 获取所有子文件夹路径
sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 提取所有子文件夹中的 class_x 数字
class_counts = {}
for sub_dir in sub_dirs:
    match = re.search(r'class_(\d+)', sub_dir)
    if match:
        class_idx = int(match.group(1))
        if class_idx in class_counts:
            class_counts[class_idx] += 1
        else:
            class_counts[class_idx] = 1

# 定义包含十个列表的大列表
class_feature_data = [[] for _ in range(10)]

def calculate_mean(Z):
    # 矩阵Z的均值，返回一个1×512的向量。
    return np.mean(Z, axis=0)

def calculate_covariance(Z, mean_Z):
    # 矩阵Z的中心化
    Z_centered = Z - mean_Z
    # 协方差矩阵 得到一个512×512的矩阵。
    return (1 / Z.shape[0]) * np.dot(Z_centered.T, Z_centered)

def process_multiple_appearances(class_idx, sub_dirs):
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
    # 该类指导客户端的样本总数
    total_samples = np.sum(sample_counts)
    # print(total_samples)
    # 该类指导客户端的总均值
    combined_mean = np.sum([sample_counts[i] * mean_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    # 聚合协方差矩阵的前半部分->累加的该类客户端数量*该类客户端的协方差矩阵
    combined_cov_matrix = np.sum([sample_counts[i] * covariance_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    # 聚合协方差矩阵的后半部分->累加的该类客户端数量*该类客户端的矩阵均值与总均值*该类客户端的矩阵均值与总均值的转置 (累加的该类客户端数量*外积矩阵)
    combined_cov_matrix += np.sum([sample_counts[i] * np.outer(mean_matrices[i] - combined_mean, mean_matrices[i] - combined_mean) for i in range(len(sample_counts))], axis=0) / total_samples

    class_feature_data[class_idx] = combined_cov_matrix

def process_single_appearance(class_idx, sub_dirs):
    # 类的指导客户端单一的情况下，聚合协方差矩阵等于协方差矩阵
    print("process_single_appearance")
    for sub_dir in sub_dirs:
        if f'class_{class_idx}' in sub_dir:
            file_path = os.path.join(sub_dir, 'final_embeddings.npy')
            Z = np.load(file_path)
            mean_Z = calculate_mean(Z)
            cov_matrix = calculate_covariance(Z, mean_Z)
            class_feature_data[class_idx] = cov_matrix

for class_idx, count in class_counts.items():
    if count >= 2:
        process_multiple_appearances(class_idx, sub_dirs)
    else:
        process_single_appearance(class_idx, sub_dirs)

# 打印并存储结果
for idx, feature_data in enumerate(class_feature_data):
    print(f"Class {idx} 数据特征:")
    np.set_printoptions(threshold=np.inf)  # 设置打印选项为显示所有值
    # print(f"协方差矩阵:\n{feature_data}\n")
    np.set_printoptions(threshold=1000)  # 恢复默认打印选项

    # 将结果保存到txt文件中
    txt_path = f'./CIFAR-10/features/alpha={alpha}_cov_matrix/class_{idx}_cov_matrix.txt'
    with open(txt_path, 'w') as f:
        f.write(str(feature_data.tolist()))

    # 将结果保存到npy文件中
    npy_path = f'./CIFAR-10/features/alpha={alpha}_cov_matrix/class_{idx}_cov_matrix.npy'
    np.save(npy_path, feature_data)
