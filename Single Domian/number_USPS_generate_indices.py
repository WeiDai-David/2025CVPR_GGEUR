# USPS是我从网上下载的mat文件，按照作者的说法是有7291个训练数据和2007个测试数据，注释脚本用来检验这一点，同时分析结构
# import scipy.io
# import os

# def check_usps_data_structure(data_dir):
#     mat_file_path = os.path.join(data_dir, 'uspsdata.mat')
#     mat = scipy.io.loadmat(mat_file_path)
#     train_data = mat['uspstrain']
#     test_data = mat['uspstest']

#     print("Train Data Structure:")
#     print(train_data)
#     print(train_data.shape)
#     print("\nTest Data Structure:")
#     print(test_data)
#     print(test_data.shape)

# if __name__ == "__main__":
#     data_dir = './data'  # 请根据实际情况修改路径
#     check_usps_data_structure(data_dir)
import os
import argparse
import numpy as np
import scipy.io

# 定义数据集加载函数
def load_usps_mat(data_dir):
    mat_file_path = os.path.join(data_dir, 'uspsdata.mat')
    mat = scipy.io.loadmat(mat_file_path)
    train_data = mat['uspstrain']
    
    # 提取图像和标签
    labels = train_data[:, 0].astype(int)
    images = train_data[:, 1:]
    
    return images, labels

# 生成索引和标签文件
def generate_index_files(dataset_name, images, labels):
    indices_per_class = {i: [] for i in range(10)}

    for idx, label in enumerate(labels):
        indices_per_class[label].append(idx)

    output_dir = f'./{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    for class_idx, indices in indices_per_class.items():
        np.save(os.path.join(output_dir, f'class_{class_idx}_indices.npy'), indices)
        np.save(os.path.join(output_dir, f'class_{class_idx}_labels.npy'), [class_idx] * len(indices))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate index files for USPS dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the USPS .mat file')
    args = parser.parse_args()

    dataset_name = 'USPS'
    images, labels = load_usps_mat(args.data_dir)
    generate_index_files(dataset_name, images, labels)
    print(f'Index files for {dataset_name} dataset have been generated.')


