import re
import torch
from PIL import Image
import numpy as np
import pickle
import os
import argparse
from torchvision import datasets, transforms, models
import torch.nn as nn

# 创建 ResNet-50 模型，并去除全连接层，然后添加一个卷积层
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        # 加载预训练的 ResNet-50 模型
        self.resnet = models.resnet50(pretrained=True)
        # 去掉最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # 添加一个新的卷积层，输出512维特征
        self.conv = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        # 对结果进行全局平均池化
        x = torch.mean(x, dim=[2, 3])
        return x

# 检查是否有CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet().to(device)
model.eval()  # 设置模型为评估模式

# 定义特征提取函数
def resnet_image_embedding(image, model):
    """
    使用 ResNet-50 模型提取图像特征，并通过自定义的 CNN 层输出512维特征

    Parameters:
    image (PIL.Image): 输入图像
    model (torch.nn.Module): 自定义的 ResNet 模型

    Returns:
    torch.Tensor: 图像的最终嵌入特征
    """
    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = preprocess(image).unsqueeze(0).to(device)  # 预处理并添加批次维度，并移到设备上

    # 执行模型前向传递
    with torch.no_grad():
        image_features = model(image)  # 提取图像特征

    return image_features.cpu()

# 解析数据集的批次文件
def unpickle(file):
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
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    elif dataset == 'CIFAR-100':
        train_file = './data/cifar-100-python/train'
        data_batch = unpickle(train_file)
        data = data_batch[b'data']
        labels = data_batch[b'fine_labels']
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    elif dataset == 'TinyImageNet':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        data, labels = zip(*[(data, label) for data, label in dataset])
        data = np.array([np.array(d).transpose(1, 2, 0) for d in data])
        labels = np.array(labels)
    return data, labels

# 加载索引文件并提取图像特征和生成标签文件
def process_indices_file(indices_file_path, output_dir, model, class_idx, data):
    indices = np.load(indices_file_path)
    all_final_embeddings = []
    labels = np.full(len(indices), class_idx)

    for index in indices:
        img_array = data[index]
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        final_embedding = resnet_image_embedding(img, model)  # 使用 ResNet 模型提取图像特征
        all_final_embeddings.append(final_embedding.squeeze(0).numpy())

    all_final_embeddings = np.array(all_final_embeddings)
    np.save(f'{output_dir}/final_embeddings.npy', all_final_embeddings)
    np.save(f'{output_dir}/labels.npy', labels)
    print(f'Features and labels for indices in {indices_file_path} saved. Total embeddings: {all_final_embeddings.shape}')

# 主函数
def main(dataset, alpha):
    base_dir = f'./{dataset}'
    indices_dir = os.path.join(base_dir, 'client_class_indices')
    output_base_dir = os.path.join(base_dir, 'features/resnet_initial')

    data, labels = load_data(dataset)

    for root, dirs, files in os.walk(indices_dir):
        for file in files:
            if file.startswith(f"alpha={alpha}") and file.endswith('_indices.npy'):
                indices_file_path = os.path.join(root, file)
                match = re.search(r'client_(\d+)_class_(\d+)_indices.npy', file)
                if match:
                    client_idx = match.group(1)
                    class_idx = int(match.group(2))
                    output_dir = os.path.join(output_base_dir, f'alpha={alpha}_class_{class_idx}_client_{client_idx}')
                    os.makedirs(output_dir, exist_ok=True)
                    print(f'Processing file: {indices_file_path}')
                    process_indices_file(indices_file_path, output_dir, model, class_idx, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10', 'CIFAR-100', 'TinyImageNet'], help='The dataset to process.')
    parser.add_argument('--alpha', type=float, default=5, help='The alpha value for Dirichlet distribution.')
    args = parser.parse_args()

    main(args.dataset, args.alpha)
