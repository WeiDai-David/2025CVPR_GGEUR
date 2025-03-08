import os
import numpy as np
import torch
from PIL import Image
import open_clip
import argparse
from torchvision import datasets, transforms
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

# Office-Home的预处理
transform_office_home = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 使用的输入大小为224x224
    # transforms.ToTensor()           # 转换为张量
])

# 加载Office-Home数据集，并根据索引提取测试集
def load_office_home_with_indices(path, domain, indices):
    """
    根据提供的索引加载Office-Home的测试集
    """
    domain_path = os.path.join(path, domain)
    dataset = datasets.ImageFolder(domain_path, transform=transform_office_home)

    # 根据提供的测试集索引提取数据
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

# 处理测试集并提取特征
def process_test_set(dataset_name, data, output_dir, model, preprocess):
    all_features = []  # 用于保存所有图像的特征
    labels = []  # 用于保存所有图像的标签

    # 使用 tqdm 包裹测试集遍历，显示进度条
    for img, label in tqdm(data, desc=f"提取 {dataset_name} 测试集特征", total=len(data)):
        
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

# 加载测试集的索引
def load_test_indices(domain, base_dir='./output_indices'):
    """
    从指定目录加载测试集的索引文件
    """
    indices_path = os.path.join(base_dir, f'{domain}', f'test_test_indices.npy')  # 修正路径拼接
    indices = np.load(indices_path)
    return indices


# 主函数
def main(datasets):
    output_base_dir = './clip_office_home_test_features'  # 保存特征的输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    data_path = './data/Office-Home'  # Office-Home数据集的路径

    for dataset_name in datasets:
        print(f"正在处理 {dataset_name} 测试集...")

        # 加载测试集的索引
        test_indices = load_test_indices(dataset_name)

        # 根据索引加载测试集
        data = load_office_home_with_indices(data_path, dataset_name, test_indices)

        output_dir = os.path.join(output_base_dir, f'{dataset_name}')
        os.makedirs(output_dir, exist_ok=True)

        # 处理整个测试集
        process_test_set(dataset_name, data, output_dir, model, preprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Art', 'Clipart', 'Product', 'Real_World'], help='要处理的测试集')
    args = parser.parse_args()

    main(args.datasets)
