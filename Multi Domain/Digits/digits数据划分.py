import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
import bz2
import scipy.io as sio

# MNIST和USPS的预处理（需要增加通道）
transform_mnist_usps = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.ToTensor(),        # 转换为张量
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 将MNIST和USPS的单通道转换为三通道
])

# SVHN和SYN的预处理（不需要增加通道）
transform_svhn_syn = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.ToTensor()         # 转换为张量
])

# MNIST加载函数
def load_mnist(path):
    train_set = datasets.MNIST(path, train=True, download=False, transform=transform_mnist_usps)
    return train_set

# USPS加载函数 
def load_usps(path):
    train_set = datasets.USPS(path, train=True, download=False, transform=transform_mnist_usps)
    return train_set

# SVHN加载函数
def load_svhn(path):
    train_set = datasets.SVHN(path, split='train', download=False, transform=transform_svhn_syn)
    return train_set

# SYN加载函数
def load_syn(path):
    train_dir = os.path.join(path, 'train')
    def load_images_from_folder(folder):
        images, labels = [], []
        for class_folder in tqdm(sorted(os.listdir(folder)), desc=f"加载 {folder}"):
            class_path = os.path.join(folder, class_folder)
            if os.path.isdir(class_path):
                for img_file in sorted(os.listdir(class_path)):
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((32, 32))  # 确保尺寸为32x32
                    images.append(np.array(img))
                    labels.append(int(class_folder))
        return images, labels
    images, labels = load_images_from_folder(train_dir)
    return list(zip(images, labels))

# 数据划分函数
def partition_data(total_data, num_clients, client_ids, percent):
    num_samples = int(total_data * percent)
    indices = np.random.permutation(total_data)
    client_indices = {}
    for i, client_id in enumerate(tqdm(client_ids, desc=f"划分数据给 {len(client_ids)} 个客户端")):
        selected_indices = np.random.choice(indices, size=num_samples, replace=False)
        client_indices[client_id] = selected_indices
        indices = np.setdiff1d(indices, selected_indices)
    return client_indices

# 保存客户端索引
def save_indices(indices_dict, domain_name, file_type="client"):
    output_dir = os.path.join('./output_indices', domain_name)
    os.makedirs(output_dir, exist_ok=True)
    for key, indices in tqdm(indices_dict.items(), desc=f"保存 {file_type} 索引"):
        txt_filename = os.path.join(output_dir, f"{file_type}_{key}_indices.txt")
        with open(txt_filename, 'w') as f:
            f.write(f"{file_type.capitalize()} {key} indices: {list(indices)}\n")
        npy_filename = os.path.join(output_dir, f"{file_type}_{key}_indices.npy")
        np.save(npy_filename, np.array(indices))

# 合并类索引
def get_combined_class_indices(dataset, total_data):
    combined_class_indices = {i: [] for i in range(10)}
    for idx in tqdm(range(total_data), desc="生成类索引"):
        label = dataset[idx][1]
        combined_class_indices[label].append(idx)
    return combined_class_indices

# 保存合并的类索引
def save_combined_class_indices(combined_class_indices, domain_name):
    output_dir = os.path.join('./output_indices', domain_name)
    os.makedirs(output_dir, exist_ok=True)
    combined_indices = {key: np.array(value) for key, value in combined_class_indices.items()}
    np.save(os.path.join(output_dir, 'combined_class_indices.npy'), combined_indices)
    with open(os.path.join(output_dir, 'combined_class_indices.txt'), 'w') as f:
        for class_label, indices in combined_class_indices.items():
            f.write(f"Class {class_label} indices: {list(indices)}\n")

# 保存客户端的类分配信息
def save_combined_client_class_distribution(domain, client_indices, class_indices, combined_report_file):
    total_class_distribution = {class_label: 0 for class_label in class_indices.keys()}

    with open(combined_report_file, 'a') as report:
        report.write(f"\n{domain} 数据集的客户端组合类分配情况:\n")
        
        for client_id, indices in client_indices.items():
            client_class_distribution = {class_label: 0 for class_label in class_indices.keys()}
            for index in indices:
                for class_label, class_idx_list in class_indices.items():
                    if index in class_idx_list:
                        client_class_distribution[class_label] += 1
                        total_class_distribution[class_label] += 1
                        break
            
            report.write(f"客户端 {client_id} 分配类数据: {client_class_distribution}\n")
            print(f"客户端 {client_id} 的类分配: {client_class_distribution}")

        report.write(f"{domain} 数据集到客户端组的类分配情况: {total_class_distribution}\n")
        print(f"{domain} 数据集到客户端组的类分配情况: {total_class_distribution}")

# 主函数
def main():
    os.makedirs('./output_indices', exist_ok=True)

    data_path = './data/Digits'
    domains = ['mnist', 'usps', 'svhn', 'syn']
    num_clients_per_domain = {'mnist': 1, 'usps': 1, 'svhn': 1, 'syn': 1}
    percent_dict = {'mnist': 0.1, 'usps': 0.1, 'svhn': 0.1, 'syn': 0.1}
    
    total_clients = 0
    client_ids_per_domain = {}

    for domain in domains:
        num_clients = num_clients_per_domain[domain]
        client_ids_per_domain[domain] = list(range(total_clients, total_clients + num_clients))
        total_clients += num_clients

    mnist_data = load_mnist(os.path.join(data_path, "MNIST"))
    usps_data = load_usps(os.path.join(data_path, "USPS"))
    svhn_data = load_svhn(os.path.join(data_path, "SVHN"))
    syn_data = load_syn(os.path.join(data_path, 'SYN'))

    datasets_dict = {'mnist': mnist_data, 'usps': usps_data, 'svhn': svhn_data, 'syn': syn_data}

    report_file = './output_indices/dataset_report.txt'
    combined_report_file = './output_indices/client_combined_class_distribution.txt'

    with open(report_file, 'w') as report:
        for domain in tqdm(domains, desc="处理每个域"):
            dataset = datasets_dict[domain]
            total_data = len(dataset)
            report.write(f"{domain} 数据集大小: {total_data} 条数据\n")

            client_indices = partition_data(total_data, num_clients_per_domain[domain], client_ids_per_domain[domain], percent_dict[domain])
            save_indices(client_indices, domain, file_type="client")

            combined_class_indices = get_combined_class_indices(dataset, total_data)
            save_combined_class_indices(combined_class_indices, domain)

            for client_id, indices in client_indices.items():
                report.write(f"客户端 {client_id} 在 {domain} 中分配了 {len(indices)} 条数据\n")

            save_combined_client_class_distribution(domain, client_indices, combined_class_indices, combined_report_file)

        print(f"已成功处理并保存所有数据集信息到 {report_file} 和 {combined_report_file} 中")

if __name__ == '__main__':
    main()
