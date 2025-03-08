import os
import numpy as np
import argparse
from tqdm import tqdm

# 保证协方差矩阵为正定矩阵，并按特征值的缩放规则进行放大
def nearest_pos_def(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 创建一个缩放因子数组，前10个特征值缩放，其他保持不变
    scale_factors = np.ones_like(eigenvalues)  # 初始化为1的缩放因子
    scale_factors[:10] = np.linspace(5, 1, 10)  # 前10个特征值从5倍递减到1倍
    
    # 对应的特征值按缩放因子进行放大
    eigenvalues = eigenvalues * scale_factors
    
    # 确保没有负特征值
    eigenvalues[eigenvalues < 0] = 0
    
    # 重新构造正定矩阵
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# 生成新样本
def generate_new_samples(feature, cov_matrix, num_generated):
    cov_matrix = nearest_pos_def(cov_matrix)
    jitter = 1e-6
    while True:
        try:
            B = np.linalg.cholesky(cov_matrix + jitter * np.eye(cov_matrix.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    new_features = np.random.multivariate_normal(feature, B @ B.T, num_generated)
    return new_features

# 扩充特征的核心函数
def expand_features_with_cov(client_id, class_idx, original_features, cov_matrices, num_per_sample, target_size):
    generated_samples = []
    for feature in original_features:
        new_samples = generate_new_samples(feature, cov_matrices[class_idx], num_per_sample)
        generated_samples.append(new_samples)
    
    # 计算需要生成的样本数量，以确保扩充后总样本量达到 target_size
    total_existing_samples = original_features.shape[0]
    num_additional_samples_needed = target_size - total_existing_samples
    all_generated_samples = np.vstack(generated_samples)
    
    if num_additional_samples_needed > 0:
        selected_indices = np.random.choice(all_generated_samples.shape[0], num_additional_samples_needed, replace=False)
        return np.vstack((original_features, all_generated_samples[selected_indices]))
    else:
        return original_features

# 执行特征扩充并保存
def process_clients(client_ids, cov_matrices, initial_features_dir, complete_features_dir, dataset_name, num_generated_per_sample, target_size):
    for client_id in client_ids:
        for class_idx in range(65):  # 修改为 65 个类别
            initial_features_path = os.path.join(initial_features_dir, f'client_{client_id}_class_{class_idx}_original_features.npy')
            if not os.path.exists(initial_features_path):
                print(f"Skipping client {client_id}, class {class_idx} - no features found.")
                continue

            # 加载原始特征
            original_features = np.load(initial_features_path)

            # 使用统一的协方差矩阵进行扩充
            print(f"Expanding client {client_id}, class {class_idx} using covariance matrix for {dataset_name}...")
            expanded_features = expand_features_with_cov(client_id, class_idx, original_features, cov_matrices, num_generated_per_sample, target_size)

            # 保存扩充后的特征
            complete_dir = os.path.join(complete_features_dir, f'client_{client_id}_class_{class_idx}')
            os.makedirs(complete_dir, exist_ok=True)
            np.save(os.path.join(complete_dir, 'final_embeddings_filled.npy'), expanded_features)

            # 为扩充的特征生成对应的标签
            labels = np.full(expanded_features.shape[0], class_idx)
            np.save(os.path.join(complete_dir, 'labels_filled.npy'), labels)

            print(f"Completed feature expansion for client {client_id}, class {class_idx}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./clip_office_home_train_features/', help='Directory containing features.')
    parser.add_argument('--cov_dir', type=str, default='./cov_matrix_output/', help='Directory containing shared covariance matrices.')
    parser.add_argument('--complete_features_dir', type=str, default='./argumented_clip_features/', help='Directory to save expanded features.')
    args = parser.parse_args()

    # 定义 Office-Home 数据集对应的客户端 ID 和生成样本的参数
    office_home_config = {
        'Art': {'client_ids': [0], 'num_generated_per_sample': 500, 'target_size': 500},
        'Clipart': {'client_ids': [1], 'num_generated_per_sample': 500, 'target_size': 500},
        'Product': {'client_ids': [2], 'num_generated_per_sample': 500, 'target_size': 500},
        'Real_World': {'client_ids': [3], 'num_generated_per_sample': 500, 'target_size': 500}
    }

    # 加载共享的协方差矩阵
    cov_matrices = {}
    for class_idx in range(65):  # 修改为 65 个类别
        cov_path = os.path.join(args.cov_dir, f'class_{class_idx}_cov_matrix.npy')
        cov_matrices[class_idx] = np.load(cov_path)

    # 对 Office-Home 数据集进行特征扩充
    for dataset_name, config in office_home_config.items():
        print(f"Processing dataset {dataset_name}...")
        process_clients(config['client_ids'], cov_matrices, os.path.join(args.base_dir, dataset_name), os.path.join(args.complete_features_dir, dataset_name), dataset_name, config['num_generated_per_sample'], config['target_size'])

if __name__ == "__main__":
    main()
