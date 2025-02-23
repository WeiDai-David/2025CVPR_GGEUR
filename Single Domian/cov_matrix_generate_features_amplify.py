# cov_matrix_generate_features.py 该脚本用聚合协方差依托原始特征为基础生成补全特征 amplify为放大版本，添加了放大特征值的参数
import numpy as np
import os
from tqdm import tqdm

# 保证协方差矩阵为正定矩阵，并放大特征值
def nearest_pos_def(cov_matrix, scale_factor=1):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 计算协方差矩阵的特征值和特征向量
    eigenvalues[eigenvalues < 0] = 0  # 将负特征值设为0，确保所有特征值非负
    eigenvalues *= scale_factor  # 放大特征值
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T  # 重新构建正定协方差矩阵

# 生成新的样本
def generate_new_samples(feature, cov_matrix, num_generated, scale_factor=1):
    cov_matrix = nearest_pos_def(cov_matrix, scale_factor)  # 确保协方差矩阵为正定并放大特征值
    jitter = 1e-6  # 初始化抖动值
    while True:
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))  # 计算协方差矩阵的Cholesky分解
            break
        except np.linalg.LinAlgError:
            jitter *= 10  # 如果Cholesky分解失败，增大抖动值重新计算

    new_features = np.random.multivariate_normal(feature, B @ B.T, num_generated)  # 使用多元正态分布生成新样本
    return new_features

# 处理每个类和客户端
def process_class_client(class_idx, client_idx, cov_matrices, initial_features_dir, complete_features_dir, scale_factor=1):
    initial_dir = os.path.join(initial_features_dir, f'alpha=0.1_class_{class_idx}_client_{client_idx}')  # 设置初始特征文件夹路径
    features_path = os.path.join(initial_dir, 'final_embeddings.npy')  # 初始特征文件路径
    if not os.path.exists(features_path):  # 如果特征文件不存在，跳过该类和客户端
        print(f"Skipping class {class_idx}, client {client_idx} because {features_path} does not exist.")
        return

    print(f"Processing class {class_idx}, client {client_idx}...")

    class_features = np.load(features_path) if os.path.exists(features_path) else np.empty((0, 512))  # 加载初始特征文件
    num_samples_current = class_features.shape[0]  # 当前样本数量
    num_samples_needed = 5000 - num_samples_current  # 需要生成的新样本数量

    if num_samples_needed > 0:
        if num_samples_needed <= num_samples_current:
            # 如果需要生成的新样本数量小于或等于当前已有样本数量
            chosen_indices = np.random.choice(num_samples_current, num_samples_needed, replace=False)
            new_samples_list = []
            for idx in chosen_indices:
                new_samples = generate_new_samples(class_features[idx], cov_matrices[class_idx], 1, scale_factor)
                new_samples_list.append(new_samples)
            new_samples_flattened = np.vstack(new_samples_list)
        else:
            # 需要生成的新样本数量大于当前已有样本数量
            num_generated_per_sample = (num_samples_needed + num_samples_current - 1) // num_samples_current  # 计算每个原始样本需要生成的新样本数量
            new_samples_list = []  # 存储所有生成的新样本
            for feature in class_features:
                new_samples = generate_new_samples(feature, cov_matrices[class_idx], num_generated_per_sample, scale_factor)  # 基于每个原始样本生成新样本
                new_samples_list.append(new_samples)
            new_samples_flattened = np.vstack(new_samples_list)  # 合并所有新样本
            # 处理生成的新样本数量与需要的新样本数量不匹配的情况
            if new_samples_flattened.shape[0] > num_samples_needed:
                new_samples_flattened = new_samples_flattened[:num_samples_needed]

        class_features = np.vstack((class_features, new_samples_flattened))  # 将新样本与原始样本合并

    complete_dir = os.path.join(complete_features_dir, f'alpha=0.1_class_{class_idx}_client_{client_idx}')  # 设置补全特征文件夹路径
    os.makedirs(complete_dir, exist_ok=True)
    np.save(os.path.join(complete_dir, 'final_embeddings_filled.npy'), class_features)  # 保存补全后的特征文件

    # 生成并保存补全特征的标签文件
    labels = np.full(class_features.shape[0], class_idx)  # 为所有补全后的样本生成对应的标签
    labels_path = os.path.join(complete_dir, 'labels_filled.npy')
    np.save(labels_path, labels)  # 保存标签文件

    print(f"Saved completed features and labels for class {class_idx}, client {client_idx}.")

def main(scale_factor=5):
    base_dir = 'C:/Users/Games/Desktop/David/MOON-main/David/CIFAR-10/features/'  # 基础目录
    cov_matrix_dir = os.path.join(base_dir, 'cov_matrix')  # 协方差矩阵文件夹路径
    initial_features_dir = os.path.join(base_dir, 'initial')  # 初始特征文件夹路径
    complete_features_dir = os.path.join(base_dir, 'complete')  # 补全特征文件夹路径

    os.makedirs(complete_features_dir, exist_ok=True)

    cov_matrices = {}
    for class_idx in tqdm(range(10), desc='Loading covariance matrices'):
        cov_matrix_path = os.path.join(cov_matrix_dir, f'class_{class_idx}_cov_matrix.npy')  # 加载协方差矩阵文件
        cov_matrices[class_idx] = np.load(cov_matrix_path)

    for class_idx in range(10):
        for client_idx in tqdm(range(10), desc=f'Processing class {class_idx}', leave=False):
            process_class_client(class_idx, client_idx, cov_matrices, initial_features_dir, complete_features_dir, scale_factor)

if __name__ == "__main__":
    main()
