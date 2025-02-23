import numpy as np


def load_npy(file_path):
    """
    加载.npy文件并返回其内容

    Parameters:
    file_path (str): .npy文件路径

    Returns:
    np.ndarray: 文件内容
    """
    return np.load(file_path)


def compare_npy(file1, file2):
    """
    比较两个.npy文件的内容是否相同

    Parameters:
    file1 (str): 第一个.npy文件路径
    file2 (str): 第二个.npy文件路径

    Returns:
    bool: 文件内容是否相同
    """
    data1 = load_npy(file1)
    data2 = load_npy(file2)

    # 转换为集合进行比较，忽略顺序
    return set(data1) == set(data2)


def main():
    dataset_name = 'CIFAR-10'  # 更改为您的数据集名称
    alpha = 0.1  # Dirichlet 分布参数
    class_idx = 0  # 示例类索引
    client_idx = 0  # 示例客户端索引

    file1 = f'./{dataset_name}/client_class_indices/alpha={alpha}_CIFAR-10_client_{client_idx}_class_{class_idx}_indices.npy'  # 替换为第一个文件的路径
    file2 = f'./{dataset_name}/best_guidance_alpha=0.1_class_0_client_0_indices.npy' # 替换为第二个文件的路径

    if compare_npy(file1, file2):
        print("两个.npy文件包含相同的索引")
    else:
        print("两个.npy文件包含不同的索引")


if __name__ == "__main__":
    main()
