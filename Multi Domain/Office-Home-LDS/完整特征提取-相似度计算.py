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

# Office-Home的预处理
transform_office_home = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 使用的输入大小为224x224
])

# 加载Office-Home数据集
def load_office_home(domain_path):
    """
    加载Office-Home数据集的所有类和样本
    """
    dataset = datasets.ImageFolder(domain_path, transform=transform_office_home)
    return dataset

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
    
    fea_num = len(all_features)
    lab_num = len(labels)
    print(f"数据集 {dataset_name} 客户端 {client_id} 类别 {class_label} 有 {fea_num}特征 {lab_num} 标签 ")
    # 保存特征和标签，使用 original_features.npy 保存训练集特征
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_original_features.npy'), all_features)
    np.save(os.path.join(output_dir, f'client_{client_id}_class_{class_label}_labels.npy'), labels)

    print(f"已保存 {dataset_name} 客户端 {client_id} 类别 {class_label} 的特征和标签到 {output_dir}，共处理 {len(all_features)} 个样本。")

# 客户端编号分配
client_range = {
    'Art': [0],
    'Clipart': [1],
    'Product': [2],
    'Real_World': [3]
}

# 主函数
def main(datasets):
    output_base_dir = './clip_office_home_all_features'  # 保存特征的输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    data_path = './data/Office-Home'  # Office-Home数据集的路径

    # 遍历数据集
    for dataset_name in datasets:
        print(f"正在处理 {dataset_name} 训练集...")

        # 获取该数据集的客户端编号
        assigned_clients = client_range[dataset_name]

        # 遍历每个客户端
        for client_id in assigned_clients:
            # 加载数据集的所有图像
            domain_path = os.path.join(data_path, dataset_name)
            dataset = load_office_home(domain_path)

            # 遍历每个类别
            for class_label in range(65):  # 假设Office-Home数据集中类别范围为0-64
                # 筛选出属于当前类别的图像
                class_data = [(img, label) for img, label in dataset if label == class_label]

                output_dir = os.path.join(output_base_dir, f'{dataset_name}')
                os.makedirs(output_dir, exist_ok=True)

                # 处理训练集
                process_train_set(dataset_name, client_id, class_label, class_data, output_dir, model, preprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Art', 'Clipart', 'Product', 'Real_World'], help='要处理的训练集')
    args = parser.parse_args()

    main(args.datasets)
