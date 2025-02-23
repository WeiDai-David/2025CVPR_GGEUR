# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # 数据集名称
# datasets = ['MNIST', 'USPS', 'SVHN', 'SYN']

# # 加载特征文件
# def load_features(dataset_name, class_idx):
#     file_path = f'./{dataset_name}/features/class_{class_idx}/class_{class_idx}_features.npy'
#     return np.load(file_path)

# # 计算相似度
# def calculate_similarity(features1, features2):
#     covariance_matrix1 = np.cov(features1, rowvar=False)
#     eigenvalues1, eigenvectors1 = np.linalg.eigh(covariance_matrix1)
#     sorted_indices1 = np.argsort(eigenvalues1)[::-1]
#     eigenvectors1 = eigenvectors1[:, sorted_indices1]

#     covariance_matrix2 = np.cov(features2, rowvar=False)
#     eigenvalues2, eigenvectors2 = np.linalg.eigh(covariance_matrix2)
#     sorted_indices2 = np.argsort(eigenvalues2)[::-1]
#     eigenvectors2 = eigenvectors2[:, sorted_indices2]

#     similarity = 0
#     for i in range(len(eigenvectors2)):
#         similarity += np.abs(np.dot(eigenvectors1[:,i].T, eigenvectors2[:,i]))

#     return similarity

# # 计算相似度矩阵
# similarity_matrix = np.zeros((len(datasets), len(datasets)))

# for i, dataset1 in enumerate(datasets):
#     features1 = load_features(dataset1, 0)
#     for j, dataset2 in enumerate(datasets):
#         features2 = load_features(dataset2, 0)
#         similarity_matrix[i, j] = calculate_similarity(features1, features2)

# # 绘制相似度矩阵图
# plt.figure(figsize=(8, 6))
# plt.imshow(similarity_matrix, cmap='Reds', interpolation='nearest')
# plt.colorbar()
# plt.xticks(range(len(datasets)), datasets, rotation=45)
# plt.yticks(range(len(datasets)), datasets)
# plt.title('Class 0 Similarity')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# 数据集名称
datasets = ['MNIST', 'USPS', 'SVHN', 'SYN']

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
    for i in range(len(eigenvectors2)):
        similarity += np.abs(np.dot(eigenvectors1[:,i].T, eigenvectors2[:,i]))

    return similarity

# 计算相似度矩阵
similarity_matrix = np.zeros((len(datasets), len(datasets)))

for i, dataset1 in enumerate(datasets):
    features1 = load_features(dataset1, 0)
    for j, dataset2 in enumerate(datasets):
        features2 = load_features(dataset2, 0)
        similarity_matrix[i, j] = calculate_similarity(features1, features2)

# 自定义颜色映射
cdict = {
    'red': [(0.0, 209/255.0, 209/255.0),
            (1.0,  97/255.0,  97/255.0)],
    'green': [(0.0, 225/255.0, 225/255.0),
              (1.0,  19/255.0,  19/255.0)],
    'blue': [(0.0, 237/255.0, 237/255.0),
             (1.0,  34/255.0,  34/255.0)]
}

custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)

# 绘制相似度矩阵图
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap=custom_cmap, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(datasets)), datasets, rotation=45)
plt.yticks(range(len(datasets)), datasets)
plt.title('Class 0 Similarity')
plt.show()

