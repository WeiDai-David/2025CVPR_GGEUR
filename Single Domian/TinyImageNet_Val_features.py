import re
import torch
from PIL import Image
import open_clip
import numpy as np
import os
import argparse
from torchvision import datasets, transforms
from tqdm import tqdm

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

# 加载 TinyImageNet 验证集数据
def load_tinyimagenet_val_data(datadir):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=f'{datadir}/new_val', transform=transform)
    data, labels = zip(*[(data, label) for data, label in dataset])
    data = np.array([np.array(d).transpose(1, 2, 0) for d in data])  # 转换为图像格式
    labels = np.array(labels)
    return data, labels

# 提取特征并生成标签文件
def extract_and_save_features(indices, labels, output_dir, model, preprocess, data):
    all_final_embeddings = []
    all_labels = []

    for idx in tqdm(indices, desc="Extracting features"):
        img_array = data[idx]
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)  # 转换为 uint8 类型
        img = Image.fromarray(img_array)  # 将数组转换为图像
        final_embedding = clip_image_embedding(img, model, preprocess)  # 提取图像特征
        all_final_embeddings.append(final_embedding.squeeze(0).numpy())
        all_labels.append(labels[idx])

    all_final_embeddings = np.array(all_final_embeddings)
    all_labels = np.array(all_labels)

    np.save(os.path.join(output_dir, 'val_final_embeddings.npy'), all_final_embeddings)
    np.save(os.path.join(output_dir, 'val_labels.npy'), all_labels)
    print(f"Features and labels for validation data saved. Total embeddings: {all_final_embeddings.shape}")

# 主函数
def main():
    datadir = './data/tiny-imagenet-200'
    index_dir = "./TinyImageNet/val_context"
    output_dir = "./TinyImageNet/val_features"
    os.makedirs(output_dir, exist_ok=True)

    data, labels = load_tinyimagenet_val_data(datadir)

    indices = np.load(os.path.join(index_dir, 'val_indices.npy'))
    labels = np.load(os.path.join(index_dir, 'val_labels.npy'))

    extract_and_save_features(indices, labels, output_dir, model, preprocess, data)

if __name__ == "__main__":
    main()
