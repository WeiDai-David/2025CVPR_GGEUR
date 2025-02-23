import os
import numpy as np
import torch
from PIL import Image
import open_clip
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms

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

# PACS 数据集的预处理
transform_pacs = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 使用的输入大小为224x224
])

# 加载 PACS 数据集，并根据索引提取训练集
def load_pacs_with_indices(path, domain, indices):
    """
    根据提供的索引加载 PACS 的训练集
    """
    domain_path = os.path.join(path, domain)
    dataset = datasets.ImageFolder(domain_path, transform=transform_pacs)

    # 根据提供的训练集索引提取数据
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

# 处理训练集并提取特征
def process_train_set(dataset_name, client_id, class_label, data, output_dir, model, preprocess):
    all_features = []  # 用于保存所有图像的特征
    labels = []  # 用于保存所有图像的标签

    # 使用 tqdm 包裹训练集遍历，显示进度条
    for img, label in tqdm(data, desc=f"提取 {dataset_name} 客户端 {client_id} 类别 {class_label} 特征", total=len(data)):
        
        # 提取特征
        feature = clip_image_embedding(img, model, preprocess)
        all_features.append(feature)
        labels.append(label)

    all_features = np.array(all_features)
    labels = np.array(labels)
    
    print(f"数据集 {dataset_name} 客户端 {client_id} 类别 {class_label} 提取了 {len(all_features)} 个特征")
    
    # 保存特征和标签，使用 original_features.npy 保存训练集特征
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), labels)

    print(f"已保存 {dataset_name} 客户端 {client_id} 类别 {class_label} 的特征和标签到 {output_dir}，共处理 {len(all_features)} 个样本。")

# 加载训练集的索引
def load_train_indices(client_id, class_label, domain, base_dir='./output_client_class_indices'):
    """
    从指定目录加载训练集的索引文件
    """
    indices_path = os.path.join(base_dir, f'{domain}', f'client_{client_id}_class_{class_label}_indices.npy')
    indices = np.load(indices_path)
    return indices

# 客户端编号分配
client_range = {
    'photo': [0],
    'art_painting': [1],
    'cartoon': [2],
    'sketch': [3]
}

# 主函数
def main(datasets):
    output_base_dir = './clip_pacs_train_features'  # 保存特征的输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    data_path = './data/PACS'  # PACS 数据集的路径

    # 遍历数据集
    for dataset_name in datasets:
        print(f"正在处理 {dataset_name} 训练集...")

        # 获取该数据集的客户端编号
        assigned_clients = client_range[dataset_name]

        # 遍历每个客户端
        for client_id in assigned_clients:
            for class_label in range(7):  # 假设 PACS 数据集有 7 个类别

                # 加载训练集的索引
                train_indices = load_train_indices(client_id, class_label, dataset_name)

                # 根据索引加载训练集
                data = load_pacs_with_indices(data_path, dataset_name, train_indices)

                output_dir = os.path.join(output_base_dir, f'{dataset_name}')
                os.makedirs(output_dir, exist_ok=True)

                # 处理训练集
                process_train_set(dataset_name, client_id, class_label, data, output_dir, model, preprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['photo', 'art_painting', 'cartoon', 'sketch'], help='要处理的训练集')
    args = parser.parse_args()

    main(args.datasets)
