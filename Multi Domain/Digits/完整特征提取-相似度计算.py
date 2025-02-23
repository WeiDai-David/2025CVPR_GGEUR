import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
import torch
import open_clip
import argparse

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 CLIP 模型和预处理函数
backbone = 'ViT-B-32'  # 使用的 CLIP 模型骨干网络
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'  # 预训练权重路径
model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
model.eval().to(device)  # 设置模型为评估模式并转移到设备

# CLIP 图像嵌入提取函数
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
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 将MNIST和USPS的单通道转换为三通道
])

# SVHN和SYN的预处理（不需要增加通道）
transform_svhn_syn = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.ToTensor()         # 转换为张量
])

# MNIST加载函数
def load_mnist(path):
    train_set = datasets.MNIST(path, train=True, download=True, transform=transform_mnist_usps)
    test_set = datasets.MNIST(path, train=False, download=False, transform=transform_mnist_usps)
    return train_set + test_set  # 合并训练集和测试集

# USPS加载函数
def load_usps(path):
    train_set = datasets.USPS(path, train=True, download=False, transform=transform_mnist_usps)
    test_set = datasets.USPS(path, train=False, download=False, transform=transform_mnist_usps)
    return train_set + test_set  # 合并训练集和测试集

# SVHN加载函数
def load_svhn(path):
    train_set = datasets.SVHN(path, split='train', download=False, transform=transform_svhn_syn)
    test_set = datasets.SVHN(path, split='test', download=False, transform=transform_svhn_syn)
    return train_set + test_set  # 合并训练集和测试集

# SYN加载函数，加载训练集和测试集
def load_syn(path):
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
    
    # 加载训练集和测试集
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'val')
    
    train_images, train_labels = load_images_from_folder(train_dir)
    test_images, test_labels = load_images_from_folder(test_dir)
    
    # 合并训练集和测试集
    all_images = train_images + test_images
    all_labels = train_labels + test_labels
    
    return list(zip(all_images, all_labels))

# 处理训练集并提取特征
def process_train_set(dataset_name, client_id, class_label, data, output_dir, model, preprocess):
    all_features = []  # 用于保存所有图像的特征
    labels = []  # 用于保存所有图像的标签

    # 使用 tqdm 包裹训练集遍历，显示进度条
    for img, label in tqdm(data, desc=f"提取 {dataset_name} 客户端 {client_id} 类别 {class_label} 特征", total=len(data)):
        # 转换为PIL图像，以便进行预处理
        
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
    
    # 打印数据集信息
    fea_num = len(all_features)
    lab_num = len(labels)
    print(f"数据集 {dataset_name} 客户端 {client_id} 类别 {class_label} 有 {fea_num}特征 {lab_num} 标签 ")
    
    # 保存特征和标签
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), labels)

    print(f"已保存 {dataset_name} 客户端 {client_id} 类别 {class_label} 的特征和标签到 {output_dir}，共处理 {len(all_features)} 个样本。")

# 客户端编号分配，根据数据集
client_range = {
    'mnist': [0],
    'usps': [1],
    'svhn': [2],
    'syn': [3]
}

# 主函数
def main(datasets):
    output_base_dir = './clip_digits_all_features'  # 保存特征的输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    dataset_loaders = {
        'mnist': load_mnist,
        'usps': load_usps,
        'svhn': load_svhn,
        'syn': load_syn
    }

    data_paths = {
        'mnist': './data/Digits/MNIST',
        'usps': './data/Digits/USPS',
        'svhn': './data/Digits/SVHN',
        'syn': './data/Digits/SYN'
    }

    # 遍历数据集
    for dataset_name in datasets:
        print(f"正在处理 {dataset_name} 数据集...")

        # 获取该数据集的客户端编号
        assigned_clients = client_range[dataset_name]

        # 遍历每个客户端
        for client_id in assigned_clients:
            # 加载数据集的所有图像
            dataset = dataset_loaders[dataset_name](data_paths[dataset_name])

            # 遍历每个类别
            for class_label in set([label for _, label in dataset]):  # 获取唯一类别标签
                class_data = [(img, label) for img, label in dataset if label == class_label]

                output_dir = os.path.join(output_base_dir, f'{dataset_name}')
                os.makedirs(output_dir, exist_ok=True)

                # 处理并提取特征
                process_train_set(dataset_name, client_id, class_label, class_data, output_dir, model, preprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'usps', 'svhn', 'syn'], help='要处理的数据集')
    args = parser.parse_args()

    main(args.datasets)
