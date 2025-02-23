# clip_features2tensor.py 该脚本用于将指导客户端集合的各个客户端数据索引文件通过clip转换成特征
# -*- coding: utf-8 -*-
import torch
from PIL import Image
import open_clip
import numpy as np
import pickle
import os

# 创建 CLIP 模型和预处理函数
backbone = 'ViT-B-32'  # 使用的 CLIP 模型骨干网络
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'  # 预训练权重路径
model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
model.eval()  # 设置模型为评估模式

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
    image = preprocess(image).unsqueeze(0)  # 预处理并添加批次维度

    # 执行模型前向传递
    with torch.no_grad():
        image_features = model.encode_image(image)  # 提取图像特征

    return image_features

# 解析 CIFAR-10 数据集的批次文件
def unpickle(file):
    """
    解压 CIFAR-10 数据集的批次文件

    Parameters:
    file (str): 批次文件路径

    Returns:
    dict: 解压后的数据字典
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载 CIFAR-10 数据集
data_batch_files = [f'C:/Users/Games/Desktop/David/MOON-main/David/data/cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)]
data_batches = [unpickle(f) for f in data_batch_files]
data = np.concatenate([batch[b'data'] for batch in data_batches])
data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
labels = np.concatenate([batch[b'labels'] for batch in data_batches])

# 加载索引文件并提取图像特征
def process_indices_file(indices_file_path, output_dir, model, preprocess):
    """
    加载索引文件并提取图像特征

    Parameters:
    indices_file_path (str): 索引文件路径
    output_dir (str): 输出目录
    model (open_clip.CLIP): CLIP 模型
    preprocess (function): 预处理函数
    """
    indices = np.load(indices_file_path)  # 加载索引文件
    all_final_embeddings = []  # 存储所有图像的最终嵌入特征

    for index in indices:
        img_array = data[index]
        img = Image.fromarray(img_array)  # 将数组转换为图像
        final_embedding = clip_image_embedding(img, model, preprocess)  # 提取图像特征
        all_final_embeddings.append(final_embedding.squeeze(0).numpy())  # 存储最终特征

    all_final_embeddings = np.array(all_final_embeddings)
    np.save(f'{output_dir}/final_embeddings.npy', all_final_embeddings)  # 保存为 .npy 文件
    np.savetxt(f'{output_dir}/final_embeddings.txt', all_final_embeddings)  # 保存为 .txt 文件
    print(f'Features for indices in {indices_file_path} saved. Total embeddings: {all_final_embeddings.shape}')

# 主函数
def main():
    """
    主函数：处理指定索引文件并提取图像特征
    """
    # 根据你的需求指定 alpha 值
    dataset_name = 'CIFAR-10'  # 可更改为 'CIFAR-100' 或 'TinyImageNet'
    alpha = 0.1  # Dirichlet 分布参数

    indices_dir = f'C:/Users/Games/Desktop/David/MOON-main/David/{dataset_name}/best_guidance_alpha={alpha}'
    output_base_dir = f'C:/Users/Games/Desktop/David/MOON-main/David/{dataset_name}/features/alpha={alpha}'

    # 遍历目录下的所有索引文件
    for root, _, files in os.walk(indices_dir):
        for file in files:
            if file.endswith('_indices.npy'):
                indices_file_path = os.path.join(root, file)
                # 解析类别索引和客户端索引
                parts = file.split('_')
                class_idx = parts[4]
                client_idx = parts[6]
                output_dir = f'{output_base_dir}_class_{class_idx}_client_{client_idx}'

                # 创建输出目录
                os.makedirs(output_dir, exist_ok=True)

                # 处理索引文件并提取特征
                process_indices_file(indices_file_path, output_dir, model, preprocess)

if __name__ == "__main__":
    main()
