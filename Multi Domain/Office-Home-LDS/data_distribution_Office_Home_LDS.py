import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from numpy.random import dirichlet

# ImageFolder_Custom类加载单个域的数据集
class ImageFolder_Custom(ImageFolder):
    def __init__(self, data_name, root, transform=None, target_transform=None, subset_train_ratio=0.7):
        super().__init__(os.path.join(root, 'Office-Home', data_name), transform=transform, target_transform=target_transform)

        self.train_index_list = []
        self.test_index_list = []

        # 计算训练集的比例
        total_samples = len(self.samples)
        train_samples = int(subset_train_ratio * total_samples)

        # 将索引随机打乱
        shuffled_indices = np.random.permutation(total_samples)

        # 前 train_samples 个用作训练集，后面的用作测试集
        self.train_index_list = shuffled_indices[:train_samples].tolist()
        self.test_index_list = shuffled_indices[train_samples:].tolist()

# 保存索引函数
def save_indices(indices_dict, domain_name, file_type):
    output_dir = os.path.join('./output_indices', domain_name)
    os.makedirs(output_dir, exist_ok=True)  # 如果输出文件夹不存在则创建

    for key, indices in tqdm(indices_dict.items(), desc=f"保存 {file_type} 索引"):
        txt_filename = os.path.join(output_dir, f"{file_type}_{key}_indices.txt")
        npy_filename = os.path.join(output_dir, f"{file_type}_{key}_indices.npy")

        # 保存为 .txt 文件
        with open(txt_filename, 'w') as f:
            f.write(f"{file_type.capitalize()} {key} indices: {list(indices)}\n")

        # 保存为 .npy 文件
        np.save(npy_filename, np.array(indices))

# 获取整个数据集的类索引
def get_class_indices(dataset):
    class_indices = {i: [] for i in range(65)}  # Office-Home 数据集有65个类
    for idx in range(len(dataset)):
        label = dataset.targets[idx]  # 获取每个样本的标签
        class_indices[label].append(idx)  # 保存整个数据集的索引到相应的类
    return class_indices

# 保存类索引（整个数据集的类索引）
def save_class_indices(class_indices, domain_name):
    output_dir = os.path.join('./output_indices', domain_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存类索引为一个文件
    txt_filename = os.path.join(output_dir, 'class_indices.txt')
    npy_filename = os.path.join(output_dir, 'class_indices.npy')

    with open(txt_filename, 'w') as f:
        for class_label, indices in class_indices.items():
            f.write(f"Class {class_label} indices: {list(indices)}\n")
    
    np.save(npy_filename, class_indices)  # 保存为.npy文件，便于后续加载

# 生成4x65的狄利克雷分布矩阵，用于划分65个类在4个客户端之间的比例
def generate_dirichlet_matrix(alpha):
    return dirichlet([alpha] * 4, 65).T  # 生成 4x65 的矩阵

# 根据狄利克雷矩阵的某一列划分样本到客户端
def split_samples_for_domain(class_train_indices, dirichlet_column):
    client_indices = []  # 每个域只有一个客户端
    class_proportions = {}  # 保存每个类的比例分配

    for class_label, indices in class_train_indices.items():
        num_samples = len(indices)
        if num_samples == 0:
            continue

        # 获取当前类的狄利克雷比例
        proportion = dirichlet_column[class_label]  # class_label作为索引，确保65个类的比例
        class_proportions[class_label] = proportion

        # 计算分配的样本数量
        num_to_allocate = int(proportion * num_samples)

        # 分配样本
        allocated_indices = indices[:num_to_allocate]
        client_indices.extend(allocated_indices)

    return client_indices, class_proportions

# 输出每个类在客户端的分配比例
def output_class_proportions(class_proportions, domain):
    print(f"\n{domain} 域的类分配比例:")
    for class_label, proportion in class_proportions.items():
        print(f"类 {class_label} 的分配比例: {proportion}")

# 主函数：处理所有域的数据集
def main(alpha=0.1):
    os.makedirs('./output_indices', exist_ok=True)
    
    data_path = './data'
    domains = ['Art', 'Clipart', 'Product', 'Real_World']  # 处理四个Office-Home数据集域
    
    # 生成 4x65 的狄利克雷分布矩阵
    dirichlet_matrix = generate_dirichlet_matrix(alpha)
    print(f"生成的 4x65 狄利克雷分布矩阵：\n{dirichlet_matrix}")

    # 输出狄利克雷矩阵每列的和
    column_sums = dirichlet_matrix.sum(axis=0)
    print(f"每列的和: {column_sums}")

    all_class_proportions = {}

    # 遍历每个域，获取训练集索引并根据狄利克雷矩阵的某一列划分样本
    for domain_index, domain in enumerate(domains):
        print(f"{domain_index}-{domain}")
        # 加载单个域的数据集，并按照自定义规则划分训练集和测试集
        dataset = ImageFolder_Custom(data_name=domain, root=data_path, transform=transforms.ToTensor())

        # 获取整个数据集的类索引，并保存
        class_indices = get_class_indices(dataset)
        save_class_indices(class_indices, domain)  # 保存为一个类索引文件

        # 划分训练集和测试集
        train_indices = dataset.train_index_list
        test_indices = dataset.test_index_list

        # 保存训练集和测试集索引
        save_indices({'train': train_indices}, domain, file_type="train")
        save_indices({'test': test_indices}, domain, file_type="test")

        # 计算每个类在训练集中的数量
        class_train_indices = {class_label: [] for class_label in class_indices.keys()}
        for idx in train_indices:
            label = dataset.targets[idx]
            class_train_indices[label].append(idx)

        # 使用狄利克雷矩阵的第 domain_index 列进行样本划分（65个类的比例）
        client_indices, class_proportions = split_samples_for_domain(class_train_indices, dirichlet_matrix[domain_index])

        # 保存每个类的分配比例
        all_class_proportions[domain] = class_proportions

        # 保存客户端的索引
        save_indices({'client': client_indices}, domain, file_type="client")

    # 输出每个类的分配比例
    for domain in domains:
        output_class_proportions(all_class_proportions[domain], domain)

    print(f"已成功处理并保存所有数据集信息")

if __name__ == '__main__':
    main(alpha=0.1)  # 可以调整alpha值来控制分配的极端程度
