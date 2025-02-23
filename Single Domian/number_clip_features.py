import re
import torch
from PIL import Image
import open_clip
import numpy as np
import os
import argparse
from torchvision import datasets, transforms

# 创建 CLIP 模型和预处理函数
backbone = 'ViT-B-32'  # 使用的 CLIP 模型骨干网络
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'  # 预训练权重路径
model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
model.eval()  # 设置模型为评估模式

# 检查是否有CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义特征提取函数
def clip_image_embedding(image, model, preprocess):
    """
    使用 CLIP 模型提取图像特征

    Parameters:
    image (PIL.Image): 输入图像
    model (open_clip.CLIP): CLIP 模型
    preprocess (function): 预处理函数

    Returns:
    torch.Tensor: 图像的最终嵌入特征
    """
    image = preprocess(image).unsqueeze(0).to(device)  # 预处理并添加批次维度，并移到设备上

    # 执行模型前向传递
    with torch.no_grad():
        image_features = model.encode_image(image)  # 提取图像特征

    return image_features.cpu()

# 加载数据集
def load_dataset(dataset_name, data_dir='./data'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'USPS':
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.USPS(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    elif dataset_name == 'SYN':
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'SYN'), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset

# 加载索引文件并提取图像特征和生成标签文件
def process_indices_file(indices_file_path, output_dir, model, preprocess, class_idx, dataset):
    """
    加载索引文件并提取图像特征

    Parameters:
    indices_file_path (str): 索引文件路径
    output_dir (str): 输出目录
    model (open_clip.CLIP): CLIP 模型
    preprocess (function): 预处理函数
    class_idx (int): 类别索引
    dataset (torchvision.datasets): 数据集
    """
    indices = np.load(indices_file_path)  # 加载索引文件
    all_final_embeddings = []  # 存储所有图像的最终嵌入特征
    labels = np.full(len(indices), class_idx)  # 生成标签

    for index in indices:
        img, _ = dataset[index]
        img = transforms.ToPILImage()(img)  # 将Tensor转换为PIL Image
        final_embedding = clip_image_embedding(img, model, preprocess)  # 提取图像特征
        all_final_embeddings.append(final_embedding.squeeze(0).numpy())  # 存储最终特征

    all_final_embeddings = np.array(all_final_embeddings)
    np.save(f'{output_dir}/class_{class_idx}_features.npy', all_final_embeddings)  # 保存为 .npy 文件
    np.save(f'{output_dir}/class_{class_idx}_labels.npy', labels)  # 保存标签为 .npy 文件
    print(f'Features and labels for indices in {indices_file_path} saved. Total embeddings: {all_final_embeddings.shape}')

# 主函数
def main(dataset_name):
    base_dir = f'./{dataset_name}'
    output_base_dir = f'./{dataset_name}/features'
    os.makedirs(output_base_dir, exist_ok=True)

    dataset = load_dataset(dataset_name)

    # 遍历所有索引文件
    for class_idx in range(10):
        indices_file_path = os.path.join(base_dir, f'class_{class_idx}_indices.npy')
        output_dir = os.path.join(output_base_dir, f'class_{class_idx}')
        os.makedirs(output_dir, exist_ok=True)
        process_indices_file(indices_file_path, output_dir, model, preprocess, class_idx, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features using CLIP for selected dataset')
    parser.add_argument('--dataset', type=str, default='SVHN', choices=['MNIST', 'USPS', 'SVHN', 'SYN'], help='Dataset to use (default: MNIST)')
    args = parser.parse_args()

    main(args.dataset)
