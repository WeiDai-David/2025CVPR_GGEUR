import os
import numpy as np
import argparse
from tqdm import tqdm

# 保证协方差矩阵为正定矩阵
def nearest_pos_def(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    scale_factors = np.ones_like(eigenvalues)
    scale_factors[:10] = np.linspace(5, 1, 10)  # 前10个特征值缩放
    eigenvalues = eigenvalues * scale_factors
    eigenvalues[eigenvalues < 0] = 0
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

# 扩充特征并选择50个补全样本 
def expand_features_with_cov(original_features, cov_matrix, num_per_sample):
    generated_samples = []
    for feature in original_features:
        new_samples = generate_new_samples(feature, cov_matrix, num_per_sample)
        generated_samples.append(new_samples)
    
    all_generated_samples = np.vstack(generated_samples)

    # 从生成的样本中随机选择50个补全样本 
    selected_indices = np.random.choice(all_generated_samples.shape[0], 50, replace=False) 
    return all_generated_samples[selected_indices]

# 合并自己的补全样本和其他客户端生成的样本，并确保原始样本保留
def combine_samples(original_features, expanded_features, other_generated_samples, target_size=50):
    num_original = original_features.shape[0]  # 原始样本数量

    # 如果原始样本数量大于目标数量，从原始样本中抽取 target_size 数量的样本
    if num_original >= target_size:
        selected_indices = np.random.choice(num_original, target_size, replace=False)
        return original_features[selected_indices]

    # 需要从生成样本中补充的数量
    num_needed = target_size - num_original

    # 合并生成的样本
    combined_generated_samples = []
    if expanded_features.shape[0] > 0:
        combined_generated_samples.append(expanded_features)
    if other_generated_samples.shape[0] > 0:
        combined_generated_samples.append(other_generated_samples)

    if len(combined_generated_samples) > 0:
        combined_generated_samples = np.vstack(combined_generated_samples)
        selected_indices = np.random.choice(combined_generated_samples.shape[0], num_needed, replace=False)
        # 合并原始样本和选择的生成样本
        final_samples = np.vstack((original_features, combined_generated_samples[selected_indices]))
    else:
        # 如果没有生成的样本，返回原始样本
        final_samples = original_features

    return final_samples

# 处理函数：每个类从原始样本生成扩充样本，并与其他客户端原型生成的样本合并
def process_clients(client_id, class_idx, original_features, cov_matrices, prototype_features, num_per_sample):
    final_samples = None

    # 1. 如果原始特征不为空，生成补全样本 
    if original_features.shape[0] > 0:
        print(f"客户端 {client_id} 类别 {class_idx} 有原始样本，执行补全样本扩充...")
        expanded_features = expand_features_with_cov(original_features, cov_matrices[class_idx], num_per_sample)
    else:
        print(f"客户端 {client_id} 类别 {class_idx} 没有原始样本，跳过原始特征扩充...")
        expanded_features = np.empty((0, cov_matrices[class_idx].shape[0]))  # 使用协方差矩阵的维度生成空数组

    # 2. 从其他客户端的类原型生成样本
    additional_samples = []
    for prototype in prototype_features:
        if prototype.size > 0:
            new_samples = generate_new_samples(prototype, cov_matrices[class_idx], 50)
            additional_samples.append(new_samples)

    if len(additional_samples) > 0:
        additional_samples = np.vstack(additional_samples)
    else:
        print(f"没有找到其他客户端的原型进行生成，跳过类 {class_idx} 的其他样本生成。")
        additional_samples = np.empty((0, cov_matrices[class_idx].shape[0]))

    # 如果原始特征为空，只返回其他客户端生成的样本
    if original_features.shape[0] == 0:
        final_samples = additional_samples[:50] if additional_samples.shape[0] >= 50 else additional_samples
    else:
        final_samples = combine_samples(original_features, expanded_features, additional_samples, target_size=50)

    return final_samples

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./clip_office_home_train_features/', help='原始特征目录')
    parser.add_argument('--cov_dir', type=str, default='./cov_matrix_output/', help='协方差矩阵目录')
    parser.add_argument('--complete_features_dir', type=str, default='./argumented_clip_features/', help='保存扩充特征的目录')
    parser.add_argument('--prototype_dir', type=str, default='./office_home_prototypes/', help='原型文件目录')
    args = parser.parse_args()

    # 定义 Office-Home 数据集对应的客户端 ID 和生成样本的参数
    datasets_config = {
        'Art': {'client_ids': [0], 'num_generated_per_sample': 50},
        'Clipart': {'client_ids': [1], 'num_generated_per_sample': 50},
        'Product': {'client_ids': [2], 'num_generated_per_sample': 50},
        'Real_World': {'client_ids': [3], 'num_generated_per_sample': 50}
    }

    # 加载协方差矩阵
    cov_matrices = {}
    for class_idx in range(65):  # 修改为65个类别
        cov_path = os.path.join(args.cov_dir, f'class_{class_idx}_cov_matrix.npy')
        cov_matrices[class_idx] = np.load(cov_path)

    # 对每个数据集进行特征扩充
    for dataset_name, config in datasets_config.items():
        for client_id in config['client_ids']:
            for class_idx in range(65):  # 修改为65个类别
                original_features_path = os.path.join(args.base_dir, dataset_name, f'client_{client_id}_class_{class_idx}_original_features.npy')
                if not os.path.exists(original_features_path):
                    print(f"跳过客户端 {client_id} 类别 {class_idx} - 未找到特征文件")
                    continue
                
                print(f"加载客户端 {client_id} 类别 {class_idx} 的原始特征...")
                original_features = np.load(original_features_path)

                # 使用其他数据集的类原型生成样本
                prototype_features = []
                other_datasets = [ds for ds in datasets_config.keys() if ds != dataset_name]  # 使用其他数据集
                
                for other_dataset in other_datasets:
                    other_client_id = datasets_config[other_dataset]['client_ids'][0]
                    prototype_path = os.path.join(args.prototype_dir, other_dataset, f'client_{other_client_id}_class_{class_idx}_prototype.npy')
                    if os.path.exists(prototype_path):
                        print(f"加载其他数据集 {other_dataset} 的原型：客户端 {other_client_id} 类别 {class_idx}")
                        prototype = np.load(prototype_path)
                        prototype_features.append(prototype)
                    else:
                        print(f"原型文件 {prototype_path} 不存在，跳过...")

                # 扩充特征并保存
                final_samples = process_clients(client_id, class_idx, original_features, cov_matrices, prototype_features, config['num_generated_per_sample'])

                complete_dir = os.path.join(args.complete_features_dir, dataset_name, f'client_{client_id}_class_{class_idx}')
                os.makedirs(complete_dir, exist_ok=True)
                np.save(os.path.join(complete_dir, 'final_embeddings_filled.npy'), final_samples)

                labels = np.full(final_samples.shape[0], class_idx)
                np.save(os.path.join(complete_dir, 'labels_filled.npy'), labels)

                print(f"完成客户端 {client_id} 类别 {class_idx} 的特征扩充并保存")

                # 在终端输出最终特征文件和标签文件的大小
                print(f"最终特征文件大小：{final_samples.shape}")
                print(f"最终标签文件大小：{labels.shape}")

if __name__ == "__main__":
    main()
