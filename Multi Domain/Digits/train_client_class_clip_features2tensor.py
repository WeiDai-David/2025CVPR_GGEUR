import os
import numpy as np
import torch
from PIL import Image
import open_clip
import argparse
from torchvision import datasets, transforms
import bz2
import scipy.io as sio
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 CLIP 模型和预处理函数
backbone = 'ViT-B-32'  # 使用的 CLIP 模型骨干网络
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'  # 预训练权重路径
model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
model.eval().to(device)  # 设置模型为评估模式并转移到设备

# 定义 CLIP 图像嵌入提取函数
def clip_image_embedding(image, model, preprocess):
    """
    使用 CLIP 模型提取图像特征
    """
    image = preprocess(image).unsqueeze(0).to(device)  # 预处理并添加批次维度
    with torch.no_grad():
        image_features = model.encode_image(image)  # 提取图像特征
    return image_features.cpu().numpy().squeeze(0)

# MNIST和USPS的预处理（需要增加通道）
transform_mnist_usps = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.ToTensor(),        # 转换为张量
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 将单通道转换为三通道
])

# SVHN和SYN的预处理（不需要增加通道）
transform_svhn_syn = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.ToTensor()         # 转换为张量
])

# MNIST加载函数
def load_mnist(path):
    path = os.path.join(path, "MNIST")
    train_set = datasets.MNIST(path, train=True, download=False, transform=transform_mnist_usps)
    return train_set

# USPS加载函数 
def load_usps(path):
    path = os.path.join(path, "USPS")
    train_set = datasets.USPS(path, train=True, download=False, transform=transform_mnist_usps)
    return train_set

# SVHN加载函数
def load_svhn(path):
    path = os.path.join(path, "SVHN")
    train_set = datasets.SVHN(path, split='train', download=False, transform=transform_svhn_syn)
    return train_set

# SYN加载函数
def load_syn(path):
    path = os.path.join(path, "SYN")
    train_dir = os.path.join(path, 'train')
    def load_images_from_folder(folder):
        images, labels = [], []
        for class_folder in tqdm(sorted(os.listdir(folder)), desc=f"加载 {folder}"):
            class_path = os.path.join(folder, class_folder)
            if os.path.isdir(class_path):
                for img_file in sorted(os.listdir(class_path)):
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((32, 32))  # 保持尺寸一致
                    images.append(np.array(img))
                    labels.append(int(class_folder))  # 标签从文件夹名称中读取
        return images, labels
    images, labels = load_images_from_folder(train_dir)
    return list(zip(images, labels))

# 解析 `dataset_report.txt` 文件，确定每个数据集对应的客户端
def parse_dataset_report(report_file):
    dataset_clients = {}
    with open(report_file, 'r') as f:
        lines = f.readlines()
        current_dataset = None
        for line in lines:
            if "数据集大小" in line:
                current_dataset = line.split()[0]
                dataset_clients[current_dataset] = []
            elif "客户端" in line:
                client_id = int(line.split()[1])
                dataset_clients[current_dataset].append(client_id)
    return dataset_clients

# 处理索引文件并提取特征
def process_client_class_indices(dataset_name, client_id, class_label, indices_file_path, output_dir, model, preprocess, data):
    indices = np.load(indices_file_path)  # 加载客户端的索引
    all_features = []  # 用于保存所有图像的特征
    labels = []  # 用于保存所有图像的标签

    for index in indices:
        img, label = data[index]  # 获取图像和标签
        
        # 处理不同格式的图像数据
        if isinstance(img, np.ndarray):  # USPS 或其他使用 numpy 数组的情况
            img = Image.fromarray(img.astype(np.uint8))  # 转换为 PIL 格式
        elif isinstance(img, torch.Tensor):  # 处理 torchvision 数据集
            img = transforms.ToPILImage()(img)
        
        # 提取特征
        feature = clip_image_embedding(img, model, preprocess)
        all_features.append(feature)
        labels.append(label)

    all_features = np.array(all_features)
    labels = np.array(labels)

    # 保存特征和标签，使用 original_features.npy 保存原始特征
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), labels)

    print(f"已保存特征和标签到 {output_dir}，共处理 {len(all_features)} 个样本。")

# 主函数
def main(datasets, report_file):
    base_dir = './output_client_class_indices'  # 保存的客户端类索引文件的目录
    output_base_dir = './clip_features'  # 保存特征的输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 解析 `dataset_report.txt` 文件，确定每个数据集对应的客户端
    dataset_clients = parse_dataset_report(report_file)
    print(f"解析的客户端数据集映射: {dataset_clients}")

    data_loaders = {
        'mnist': load_mnist,
        'usps': load_usps,
        'svhn': load_svhn,
        'syn': load_syn
    }

    for dataset_name in datasets:
        if dataset_name in data_loaders:
            data = data_loaders[dataset_name]('./data/Digits')  # 使用一致的数据加载函数
            print(f"正在处理 {dataset_name} 数据集...")

            # 根据 `dataset_report.txt` 中的客户端分配，处理每个客户端
            if dataset_name in dataset_clients:
                clients = dataset_clients[dataset_name]
                print(f"{dataset_name} 数据集的客户端: {clients}")
                for client_id in clients:
                    for class_label in range(10):  # 假设类别范围为0-9
                        indices_file_path = os.path.join(base_dir, f'{dataset_name}', f'client_{client_id}_class_{class_label}_indices.npy')
                        if os.path.exists(indices_file_path):
                            output_dir = os.path.join(output_base_dir, f'{dataset_name}')
                            os.makedirs(output_dir, exist_ok=True)

                            print(f"处理客户端 {client_id} 类别 {class_label} 的索引文件: {indices_file_path}")
                            process_client_class_indices(dataset_name, client_id, class_label, indices_file_path, output_dir, model, preprocess, data)
                        else:
                            print(f"客户端 {client_id} 类别 {class_label} 的索引文件 {indices_file_path} 不存在，跳过...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'usps', 'svhn', 'syn'], help='要处理的数据集')
    parser.add_argument('--report_file', type=str, default='./output_indices/dataset_report.txt', help='数据集和客户端的映射文件')
    args = parser.parse_args()

    main(args.datasets, args.report_file)
