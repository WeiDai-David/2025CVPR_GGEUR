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
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 将MNIST和USPS的单通道转换为三通道
])

# SVHN和SYN的预处理（不需要增加通道）
transform_svhn_syn = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.ToTensor()         # 转换为张量
])

# MNIST加载函数
def load_mnist(path):
    path = os.path.join(path, "MNIST")
    test_set = datasets.MNIST(path, train=False, download=False, transform=transform_mnist_usps)
    return test_set

# USPS加载函数
def load_usps(path):
    path = os.path.join(path, "USPS")
    test_set = datasets.USPS(path, train=False, download=False, transform=transform_mnist_usps)
    return test_set

# SVHN加载函数
def load_svhn(path):
    path = os.path.join(path, "SVHN")
    test_set = datasets.SVHN(path, split='test', download=False, transform=transform_svhn_syn)
    return test_set

# SYN加载函数
def load_syn(path):
    path = os.path.join(path, "SYN")
    test_dir = os.path.join(path, 'val')
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
    images, labels = load_images_from_folder(test_dir)
    return list(zip(images, labels))

# 处理测试集并提取特征
def process_test_set(dataset_name, data, output_dir, model, preprocess):
    all_features = []  # 用于保存所有图像的特征
    labels = []  # 用于保存所有图像的标签

    # 使用 tqdm 包裹测试集遍历，显示进度条
    for img, label in tqdm(data, desc=f"提取 {dataset_name} 测试集特征", total=len(data)):
        
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

    # 保存特征和标签，使用 test_features.npy 保存测试集特征
    np.save(os.path.join(output_dir, f'{dataset_name}_test_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'{dataset_name}_test_labels.npy'), labels)

    print(f"已保存 {dataset_name} 测试集特征和标签到 {output_dir}，共处理 {len(all_features)} 个样本。")

# 主函数
def main(datasets):
    output_base_dir = './clip_test_features'  # 保存特征的输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 数据集加载器字典
    data_loaders = {
        'mnist': load_mnist,
        'usps': load_usps,
        'svhn': load_svhn,
        'syn': load_syn
    }

    for dataset_name in datasets:
        if dataset_name in data_loaders:
            data = data_loaders[dataset_name]('./data/Digits')  # 使用一致的数据加载函数
            print(f"正在处理 {dataset_name} 测试集...")

            output_dir = os.path.join(output_base_dir, f'{dataset_name}')
            os.makedirs(output_dir, exist_ok=True)

            # 处理整个测试集
            process_test_set(dataset_name, data, output_dir, model, preprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'usps', 'svhn', 'syn'], help='要处理的测试集')
    args = parser.parse_args()

    main(args.datasets)
