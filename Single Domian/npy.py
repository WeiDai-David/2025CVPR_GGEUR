import numpy as np
import os


# 定义加载和打印 .npy 文件的函数
def print_npy_file(file_path):
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"Contents of {file_path}:")
        print(data)
        print(data.shape)
    else:
        print(f"File {file_path} does not exist.")


def main():
    # 示例 .npy 文件路径
    dataset_name = 'CIFAR-100'  # 更改为您的数据集名称

    # 构建文件路径
    # file_path = f'./{dataset_name}/features/resnet/guide/alpha=0.1_class_1_client_9/final_embeddings.npy'
    file_path = f'./{dataset_name}/features/alpha=0.5_test/labels.npy'
    # 打印 .npy 文件内容
    print_npy_file(file_path)


if __name__ == "__main__":
    main()
