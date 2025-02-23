import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 从文件加载特征
def load_features(dataset_name, client_id, class_label):
    feature_path = f'./clip_office_home_all_features/{dataset_name}/client_{client_id}_class_{class_label}_original_features.npy'
    return np.load(feature_path)

# 计算两个数据矩阵之间的相似度，返回相似度值。
def compute_similarity_matrix(data_matrix1, data_matrix2):
    # 计算协方差矩阵
    covariance_matrix1 = np.cov(data_matrix1, rowvar=False)
    covariance_matrix2 = np.cov(data_matrix2, rowvar=False)

    # 对协方差矩阵进行特征值分解
    eigenvalues1, eigenvectors1 = np.linalg.eigh(covariance_matrix1)
    eigenvalues2, eigenvectors2 = np.linalg.eigh(covariance_matrix2)

    # 对特征值排序
    sorted_indices1 = np.argsort(eigenvalues1)[::-1]
    eigenvalues1 = eigenvalues1[sorted_indices1]
    eigenvectors1 = eigenvectors1[:, sorted_indices1]

    sorted_indices2 = np.argsort(eigenvalues2)[::-1]
    eigenvalues2 = eigenvalues2[sorted_indices2]
    eigenvectors2 = eigenvectors2[:, sorted_indices2]

    # 计算相似度
    similarity = 0
    for i in range(3):
        similarity += np.abs(np.dot(eigenvectors1[:, i].T, eigenvectors2[:, i]))

    return similarity

# 计算两个数据集之间所有类的相似度矩阵
def compute_similarity_for_datasets(dataset_name1, client_id1, dataset_name2, client_id2):
    similarity_matrix = np.zeros((65, 65))  # Office-Home数据集有65个类别
    for i in range(65):
        for j in range(65):
            data1 = load_features(dataset_name1, client_id1, i)
            data2 = load_features(dataset_name2, client_id2, j)
            similarity_matrix[i, j] = compute_similarity_matrix(data1, data2)
    return similarity_matrix

# 生成热力图并保存
def save_heatmap(similarity_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# 保存相似度矩阵为txt文件
def save_similarity_matrix(matrix, filename):
    np.savetxt(filename, matrix, fmt="%.6f")

# 客户端编号分配
client_range = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}

# 主函数：处理不同数据集组合，并保存结果
def process_and_save(dataset_name1, dataset_name2, output_dir):
    client_id1 = client_range[dataset_name1][0]
    client_id2 = client_range[dataset_name2][0]

    # 计算相似度矩阵
    similarity_matrix = compute_similarity_for_datasets(dataset_name1, client_id1, dataset_name2, client_id2)

    # 保存相似度矩阵为txt文件
    txt_filename = os.path.join(output_dir, f"{dataset_name1}_vs_{dataset_name2}_similarity_matrix.txt")
    save_similarity_matrix(similarity_matrix, txt_filename)

    # 生成并保存热力图
    heatmap_filename = os.path.join(output_dir, f"{dataset_name1}_vs_{dataset_name2}_heatmap.png")
    save_heatmap(similarity_matrix, f"{dataset_name1} vs {dataset_name2} Similarity", heatmap_filename)

# 设置输出目录
output_dir = './output_similarity_results'
os.makedirs(output_dir, exist_ok=True)

# 处理Art和Clipart组合
process_and_save('Art', 'Clipart', output_dir)

# 处理Art和Product组合
process_and_save('Art', 'Product', output_dir)

# 处理Art和Real_World组合
process_and_save('Art', 'Real_World', output_dir)

# 处理Clipart和Product组合
process_and_save('Clipart', 'Product', output_dir)

# 处理Clipart和Real_World组合
process_and_save('Clipart', 'Real_World', output_dir)

# 处理Product和Real_World组合
process_and_save('Product', 'Real_World', output_dir)

print("处理完成，已保存相似度矩阵和热力图。")
