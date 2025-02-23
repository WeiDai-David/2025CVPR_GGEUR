import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# 数据集名称
dataset1 = 'MNIST'
dataset2 = 'SVHN'

# 加载特征文件
def load_features(dataset_name, class_idx):
    file_path = f'./{dataset_name}/features/class_{class_idx}/class_{class_idx}_features.npy'
    return np.load(file_path)

# 计算相似度
def calculate_similarity(features1, features2):
    covariance_matrix1 = np.cov(features1, rowvar=False)
    eigenvalues1, eigenvectors1 = np.linalg.eigh(covariance_matrix1)
    sorted_indices1 = np.argsort(eigenvalues1)[::-1]
    eigenvectors1 = eigenvectors1[:, sorted_indices1]

    covariance_matrix2 = np.cov(features2, rowvar=False)
    eigenvalues2, eigenvectors2 = np.linalg.eigh(covariance_matrix2)
    sorted_indices2 = np.argsort(eigenvalues2)[::-1]
    eigenvectors2 = eigenvectors2[:, sorted_indices2]

    similarity = 0
    for i in range(5):
        similarity += np.abs(np.dot(eigenvectors1[:,i].T, eigenvectors2[:,i]))

    return similarity

# 计算两个数据集之间的相似度矩阵
def calculate_similarity_matrix(dataset1, dataset2):
    similarity_matrix = np.zeros((10, 10))

    for i in range(10):
        features1 = load_features(dataset1, i)
        for j in range(10):
            features2 = load_features(dataset2, j)
            similarity_matrix[i, j] = calculate_similarity(features1, features2)

    return similarity_matrix

# 计算相似度矩阵
similarity_matrix = calculate_similarity_matrix(dataset1, dataset2)

# 自定义颜色映射，从白色到红色
cdict = {
    'red': [(0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0)],
    'green': [(0.0, 1.0, 1.0),
              (1.0, 0.0, 0.0)],
    'blue': [(0.0, 1.0, 1.0),
             (1.0, 0.0, 0.0)]
}

custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)

# 创建输出目录
output_dir = './number_image'
os.makedirs(output_dir, exist_ok=True)

# 绘制相似度矩阵图并保存
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap=custom_cmap, interpolation='nearest')
plt.colorbar()
plt.xticks(range(10), range(10))
plt.yticks(range(10), range(10))
plt.xlabel(f'{dataset1} Classes')
plt.ylabel(f'{dataset2} Classes')
plt.title(f'{dataset1} vs {dataset2} Class Similarity')
plt.savefig(os.path.join(output_dir, f'5{dataset1}-{dataset2}_class_similarity.png'))
plt.show()
