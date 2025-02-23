import re
import torch
from PIL import Image
import open_clip
import numpy as np
import pickle
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

# 解析数据集的批次文件
def unpickle(file):
    """
    解压数据集的批次文件

    Parameters:
    file (str): 批次文件路径

    Returns:
    dict: 解压后的数据字典
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载数据集
def load_data(dataset):
    if dataset == 'CIFAR-10':
        data_batch_files = [f'./data/cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)]
        data_batches = [unpickle(f) for f in data_batch_files]
        data = np.concatenate([batch[b'data'] for batch in data_batches])
        labels = np.concatenate([batch[b'labels'] for batch in data_batches])
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
    elif dataset == 'CIFAR-100':
        train_file = './data/cifar-100-python/train'
        data_batch = unpickle(train_file)
        data = data_batch[b'data']
        labels = data_batch[b'fine_labels']
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
    elif dataset == 'TinyImageNet':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        data, labels = zip(*[(data, label) for data, label in dataset])
        data = np.array([np.array(d).transpose(1, 2, 0) for d in data])  # 转换为图像格式
        labels = np.array(labels)
    return data, labels

# 加载索引文件并提取图像特征和生成标签文件
def process_indices_file(indices_file_path, output_dir, model, preprocess, class_idx, data):
    """
    加载索引文件并提取图像特征

    Parameters:
    indices_file_path (str): 索引文件路径
    output_dir (str): 输出目录
    model (open_clip.CLIP): CLIP 模型
    preprocess (function): 预处理函数
    class_idx (int): 类别索引
    data (numpy.ndarray): 图像数据
    """
    indices = np.load(indices_file_path)  # 加载索引文件
    all_final_embeddings = []  # 存储所有图像的最终嵌入特征
    labels = np.full(len(indices), class_idx)  # 生成标签

    for index in indices:
        img_array = data[index]
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)  # 转换为 uint8 类型
        img = Image.fromarray(img_array)  # 将数组转换为图像
        final_embedding = clip_image_embedding(img, model, preprocess)  # 提取图像特征
        all_final_embeddings.append(final_embedding.squeeze(0).numpy())  # 存储最终特征

    all_final_embeddings = np.array(all_final_embeddings)
    np.save(f'{output_dir}/final_embeddings.npy', all_final_embeddings)  # 保存为 .npy 文件
    np.save(f'{output_dir}/labels.npy', labels)  # 保存标签为 .npy 文件
    print(f'Features and labels for indices in {indices_file_path} saved. Total embeddings: {all_final_embeddings.shape}')

# 主函数
def main(dataset, alpha):
    """
    主函数：处理所有索引文件并提取图像特征
    """
    # 设置基本路径和输出路径
    base_dir = f'./{dataset}'
    indices_dir = os.path.join(base_dir, 'client_class_indices')
    output_base_dir = os.path.join(base_dir, 'features/initial')

    data, labels = load_data(dataset)

    # 遍历所有索引文件
    for root, dirs, files in os.walk(indices_dir):
        for file in files:
            if file.startswith(f"alpha={alpha}"):
                if file.endswith('_indices.npy'):
                    indices_file_path = os.path.join(root, file)
                    # 从文件名中提取类和客户端信息
                    match = re.search(r'client_(\d+)_class_(\d+)_indices.npy', file)
                    if match:
                        client_idx = match.group(1)
                        class_idx = int(match.group(2))
                        output_dir = os.path.join(output_base_dir, f'alpha={alpha}_class_{class_idx}_client_{client_idx}')
                        os.makedirs(output_dir, exist_ok=True)
                        print(f'Processing file: {indices_file_path}')  # 添加调试信息
                        process_indices_file(indices_file_path, output_dir, model, preprocess, class_idx, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TinyImageNet', choices=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'], help='The dataset to process.')
    parser.add_argument('--alpha', type=float, default=0.2, help='The alpha value for Dirichlet distribution.')
    args = parser.parse_args()

    main(args.dataset, args.alpha)
