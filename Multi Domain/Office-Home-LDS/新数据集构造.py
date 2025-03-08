import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from numpy.random import dirichlet
import shutil


# ImageFolder_Custom类加载单个域的数据集
class ImageFolder_Custom(ImageFolder):
    def __init__(self, data_name, root, transform=None, target_transform=None, subset_train_ratio=0.7):
        super().__init__(os.path.join(root, 'Office-Home', data_name), transform=transform,
                         target_transform=target_transform)

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


# 根据索引复制图片并重命名
def copy_images(dataset, indices, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for idx in tqdm(indices, desc=f"复制到 {target_dir}"):
        source_path, label = dataset.samples[idx]

        # 生成唯一文件名（根据类标签和索引）
        new_filename = f"class_{label}_index_{idx}.jpg"
        target_path = os.path.join(target_dir, new_filename)

        # 复制图片
        shutil.copy(source_path, target_path)


# 构建新数据集
def construct_new_dataset(dataset, train_indices, test_indices, client_indices, domain, alpha):
    base_path = f'./new_dataset/Office-Home-{alpha}/{domain}'
    os.makedirs(base_path, exist_ok=True)

    # 复制训练集和测试集
    copy_images(dataset, train_indices, os.path.join(base_path, 'train'))
    copy_images(dataset, test_indices, os.path.join(base_path, 'test'))

    # 复制客户端数据集
    client_path = os.path.join(base_path, 'client')
    copy_images(dataset, client_indices, client_path)


# 保存索引函数
def save_indices(indices_dict, domain_name, file_type, alpha):
    output_dir = os.path.join(f'./new_dataset/Office-Home-{alpha}/output_indices', domain_name)
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
def save_class_indices(class_indices, domain_name, alpha):
    output_dir = os.path.join(f'./new_dataset/Office-Home-{alpha}/output_indices', domain_name)
    os.makedirs(output_dir, exist_ok=True)

    txt_filename = os.path.join(output_dir, 'class_indices.txt')
    npy_filename = os.path.join(output_dir, 'class_indices.npy')

    with open(txt_filename, 'w') as f:
        for class_label, indices in class_indices.items():
            f.write(f"Class {class_label} indices: {list(indices)}\n")

    np.save(npy_filename, class_indices)


# 生成4x65的狄利克雷分布矩阵，用于划分65个类在4个客户端之间的比例
def generate_dirichlet_matrix(alpha):
    return dirichlet([alpha] * 4, 65).T  # 生成 4x65 的矩阵


# 根据狄利克雷矩阵的某一列划分样本到客户端
def split_samples_for_domain(class_train_indices, dirichlet_column):
    client_indices = []
    class_proportions = {}

    for class_label, indices in class_train_indices.items():
        num_samples = len(indices)
        if num_samples == 0:
            continue

        # 获取当前类的狄利克雷比例
        proportion = dirichlet_column[class_label]
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

# 保存所有域的类分配数量到一个文件
def save_class_allocation_combined(domains, alpha):

    output_dir = f'./new_dataset/Office-Home-{alpha}/output_indices'
    combined_allocation = []

    # 遍历每个域
    for domain_name in domains:
        domain_output_dir = os.path.join(output_dir, domain_name)
        class_indices_path = os.path.join(domain_output_dir, 'class_indices.npy')
        client_indices_path = os.path.join(domain_output_dir, 'client_client_indices.npy')

        # 确保文件存在
        if not os.path.exists(class_indices_path) or not os.path.exists(client_indices_path):
            print(f"文件缺失: {class_indices_path} 或 {client_indices_path}")
            continue

        # 加载npy文件
        class_indices = np.load(class_indices_path, allow_pickle=True).item()
        client_indices = np.load(client_indices_path)

        # 初始化当前域的类分配
        domain_class_allocation = {class_label: 0 for class_label in class_indices.keys()}

        # 统计每个类的样本数量
        for idx in client_indices:
            for class_label, indices in class_indices.items():
                if idx in indices:
                    domain_class_allocation[class_label] += 1
                    break

        # 格式化当前域的类分配信息
        allocation_str = f"{domain_name}[" + ",".join(f"{class_label}:{count}" for class_label, count in domain_class_allocation.items()) + "]"
        combined_allocation.append(allocation_str)

    # 保存所有域的类分配信息到一个txt文件
    combined_txt_filename = os.path.join(output_dir, 'combined_class_allocation.txt')
    with open(combined_txt_filename, 'w') as f:
        for allocation in combined_allocation:
            f.write(f"{allocation}\n")
    print(f"已保存所有域的类分配数量到 {combined_txt_filename}")


# 主函数：处理所有域的数据集
def main(alpha):
    os.makedirs(f'./new_dataset/Office-Home-{alpha}/output_indices', exist_ok=True)

    data_path = './data'
    domains = ['Art', 'Clipart', 'Product', 'Real World']

    dirichlet_matrix = generate_dirichlet_matrix(alpha)
    print(f"生成的 4x65 狄利克雷分布矩阵：\n{dirichlet_matrix}")

    all_class_proportions = {}

    for domain_index, domain in enumerate(domains):
        print(f"{domain_index}-{domain}")

        dataset = ImageFolder_Custom(data_name=domain, root=data_path, transform=transforms.ToTensor())

        class_indices = get_class_indices(dataset)
        save_class_indices(class_indices, domain, alpha)

        train_indices = dataset.train_index_list
        test_indices = dataset.test_index_list

        save_indices({'train': train_indices}, domain, file_type="train", alpha=alpha)
        save_indices({'test': test_indices}, domain, file_type="test", alpha=alpha)

        class_train_indices = {class_label: [] for class_label in class_indices.keys()}
        for idx in train_indices:
            label = dataset.targets[idx]
            class_train_indices[label].append(idx)

        client_indices, class_proportions = split_samples_for_domain(class_train_indices,
                                                                     dirichlet_matrix[domain_index])

        all_class_proportions[domain] = class_proportions
        save_indices({'client': client_indices}, domain, file_type="client", alpha=alpha)

        # 调用构建新数据集的函数
        construct_new_dataset(dataset, train_indices, test_indices, client_indices, domain, alpha)

    for domain in domains:
        output_class_proportions(all_class_proportions[domain], domain)

    print(f"已成功处理并保存所有数据集信息")

    save_class_allocation_combined(domains, alpha)

if __name__ == '__main__':
    main(alpha=0.1)
