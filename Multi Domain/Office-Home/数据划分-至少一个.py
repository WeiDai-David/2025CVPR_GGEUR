import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

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

# 保存客户端类分配情况到同一个文件
def save_combined_client_distribution(report_filename, domain_name, client_indices, class_indices):
    # 汇总各客户端的类分配信息
    client_distribution = {}
    total_class_distribution = {class_label: 0 for class_label in class_indices.keys()}
    
    for client_id, indices in client_indices.items():
        class_distribution = {class_label: 0 for class_label in class_indices.keys()}
        for idx in indices:
            for class_label, class_idx_list in class_indices.items():
                if idx in class_idx_list:
                    class_distribution[class_label] += 1
                    total_class_distribution[class_label] += 1
                    break
        client_distribution[client_id] = class_distribution
    
    # 追加写入到同一个文件
    with open(report_filename, 'a') as report_file:
        report_file.write(f"\n{domain_name} 数据集的客户端组合类分配情况:\n")
        
        for client_id, class_dist in client_distribution.items():
            report_file.write(f"客户端 {client_id} 分配类数据: {class_dist}\n")
        
        report_file.write(f"{domain_name} 数据集到客户端组的类分配情况: {total_class_distribution}\n")
    print(f"已成功保存 {domain_name} 的客户端类分配情况到 {report_filename}")

# 主函数：处理所有域的数据集
def main():
    os.makedirs('./output_indices', exist_ok=True)
    
    data_path = './data'
    domains = ['Art', 'Clipart', 'Product', 'Real_World']  # 处理四个Office-Home数据集域
    
    # 客户端编号分配，根据数据集顺序
    client_range = {
        'Art': [0],
        'Clipart': [1],
        'Product': [2],
        'Real_World': [3]
    }

    percent_dict = {'Art': 0.05, 'Clipart': 0.05, 'Product': 0.05, 'Real_World': 0.05}

    # 定义报告文件路径
    combined_report_file = './output_indices/client_combined_class_distribution.txt'

    # 清空或创建报告文件
    open(combined_report_file, 'w').close()

    for domain in domains:
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

        # 按比例为客户端分配数据
        assigned_clients = client_range[domain]  # 根据domain获取对应的客户端编号范围
        client_indices = {client_id: [] for client_id in assigned_clients}

        # 客户端样本数量是按整个数据集的比例来确定的
        total_samples = len(dataset.samples)
        client_sample_size = int(total_samples * percent_dict[domain])

        # 从训练集中抽取客户端的样本
        selected_train_indices = np.random.choice(train_indices, client_sample_size, replace=False)

        # 这里为每个客户端分配样本
        for client_id in assigned_clients:
            client_indices[client_id] = selected_train_indices.tolist()  # 全部分配给客户端

            # 检查客户端中每个类的样本数量，如果为0则从训练集中抽取一个
            for class_label in class_indices.keys():
                if class_train_indices[class_label] and not any(idx in client_indices[client_id] for idx in class_train_indices[class_label]):
                    # 从训练集中抽取一个样本
                    additional_sample = np.random.choice(class_train_indices[class_label], 1)
                    client_indices[client_id].extend(additional_sample)
                    print(f"从训练集中为客户端 {client_id} 的类 {class_label} 抽取了一个额外样本")

        # 保存客户端的索引
        save_indices(client_indices, domain, file_type="client")

        # 保存客户端的类分配情况到同一个文件
        save_combined_client_distribution(combined_report_file, domain, client_indices, class_indices)

    print(f"已成功处理并保存所有数据集信息到 {combined_report_file}")


if __name__ == '__main__':
    main()
