import numpy as np
import os

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

def process_class_aggregated_covariance(class_idx, all_client_feature_paths):
    """
    处理所有客户端的类特征，计算聚合协方差矩阵
    """
    print(f"Processing class {class_idx} across all clients.")
    sample_counts = []
    mean_matrices = []
    covariance_matrices = []

    for feature_file in all_client_feature_paths:
        print(f"Loading features from: {feature_file}")
        Z = np.load(feature_file)

        # 如果样本数量为0，跳过该客户端
        if Z.shape[0] == 0:
            print(f"Skipping {feature_file} due to 0 samples.")
            continue
        
        mean_Z = calculate_mean(Z)
        cov_matrix = calculate_covariance(Z, mean_Z)
        sample_counts.append(Z.shape[0])
        mean_matrices.append(mean_Z)
        covariance_matrices.append(cov_matrix)

    # 如果所有客户端都没有样本，返回空的协方差矩阵
    if len(sample_counts) == 0:
        print(f"No valid samples for class {class_idx}. Returning an empty covariance matrix.")
        return np.zeros((512, 512))

    # 计算聚合均值和协方差矩阵
    total_samples = np.sum(sample_counts)
    combined_mean = np.sum([sample_counts[i] * mean_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    combined_cov_matrix = np.sum([sample_counts[i] * covariance_matrices[i] for i in range(len(sample_counts))], axis=0) / total_samples
    combined_cov_matrix += np.sum([sample_counts[i] * np.outer(mean_matrices[i] - combined_mean, mean_matrices[i] - combined_mean) for i in range(len(sample_counts))], axis=0) / total_samples

    return combined_cov_matrix

def main():
    """
    主函数：处理所有客户端的特征文件并提取每个类的聚合协方差矩阵
    """
    base_dir = './clip_office_home_train_features/'  # 包含客户端特征的文件夹路径
    output_cov_dir = './cov_matrix_output/'  # 保存协方差矩阵的输出目录
    os.makedirs(output_cov_dir, exist_ok=True)

    # 已知客户端编号分配
    client_range = {
        'Art': [0],
        'Clipart': [1],
        'Product': [2],
        'Real_World': [3]
    }

    num_classes = 65  # 修改为65个类，适应Office-Home
    class_feature_data = [np.zeros((512, 512)) for _ in range(num_classes)]  # 初始化每个类的协方差矩阵

    # 准备所有客户端的类特征路径
    for class_idx in range(num_classes):
        all_client_feature_paths = []  # 存储所有客户端的类特征文件路径
        for dataset_name, clients in client_range.items():
            for client_id in clients:
                feature_file = os.path.join(base_dir, f'{dataset_name}', f'client_{client_id}_class_{class_idx}_original_features.npy')
                if os.path.exists(feature_file):
                    all_client_feature_paths.append(feature_file)

        # 处理每个类，计算聚合协方差矩阵
        class_feature_data[class_idx] = process_class_aggregated_covariance(class_idx, all_client_feature_paths)

    # 保存每个类的聚合协方差矩阵
    for idx, feature_data in enumerate(class_feature_data):
        print(f"Class {idx} 数据特征聚合完成.")
        txt_path = f'{output_cov_dir}/class_{idx}_cov_matrix.txt'
        with open(txt_path, 'w') as f:
            f.write(str(feature_data.tolist()))

        npy_path = f'{output_cov_dir}/class_{idx}_cov_matrix.npy'
        np.save(npy_path, feature_data)

    print("所有聚合协方差矩阵保存成功。")

if __name__ == "__main__":
    main()
