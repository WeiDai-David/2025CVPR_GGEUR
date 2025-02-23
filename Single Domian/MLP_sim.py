import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import open_clip
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载所有类的原始特征文件和补全后的特征文件
def load_features_and_labels(client_idx, base_dir='C:/Users/Games/Desktop/David/MOON-main/David/CIFAR-10/features'):
    original_features_list = []
    augmented_features_list = []
    original_labels_list = []
    augmented_labels_list = []

    for class_idx in range(10):
        original_features_path = os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'final_embeddings.npy')
        augmented_features_path = os.path.join(base_dir, 'complete', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'final_embeddings_filled.npy')
        original_labels_path = os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels.npy')
        augmented_labels_path = os.path.join(base_dir, 'complete', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')

        if os.path.exists(original_features_path):
            original_features = np.load(original_features_path)
            original_labels = np.load(original_labels_path)
            original_features_list.append(torch.tensor(original_features, dtype=torch.float32))
            original_labels_list.append(torch.tensor(original_labels, dtype=torch.long))

        if os.path.exists(augmented_features_path):
            augmented_features = np.load(augmented_features_path)
            augmented_labels = np.load(augmented_labels_path)
            augmented_features_list.append(torch.tensor(augmented_features, dtype=torch.float32))
            augmented_labels_list.append(torch.tensor(augmented_labels, dtype=torch.long))

    return original_features_list, augmented_features_list, original_labels_list, augmented_labels_list

# 转换为PyTorch张量
original_features_list, augmented_features_list, original_labels_list, augmented_labels_list = load_features_and_labels(client_idx=0)

# 将原始特征和补全特征分别合并成单一的数据集
original_dataset = ConcatDataset([TensorDataset(features, labels) for features, labels in zip(original_features_list, original_labels_list)])
augmented_dataset = ConcatDataset([TensorDataset(features, labels) for features, labels in zip(augmented_features_list, augmented_labels_list)])

# 创建数据加载器
def create_data_loader(dataset, batch_size=64):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

original_train_loader = create_data_loader(original_dataset)
augmented_train_loader = create_data_loader(augmented_dataset)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 初始化模型、损失函数和优化器
input_dim = original_features_list[0].shape[1]
hidden_dim = 256
output_dim = 10  # CIFAR-10 有 10 个类别，输出维度设置为 10

def train_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 加载 CIFAR-10 测试数据
def load_cifar10_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = datasets.CIFAR10(root='C:/Users/Games/Desktop/David/MOON-main/David/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

cifar10_test_loader = load_cifar10_data()

# 定义特征提取函数
def extract_image_features(model, preprocess, loader):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            images = torch.stack([preprocess(Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))).to(device) for img in images])
            features = model.encode_image(images)
            features_list.append(features.cpu())
            labels_list.append(labels)
    return torch.cat(features_list), torch.cat(labels_list)

# 加载预训练的 CLIP 模型
backbone = 'ViT-B-32'  # 使用的 CLIP 模型骨干网络
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'  # 预训练权重路径
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
clip_model.eval().to(device)  # 设置模型为评估模式并转移到 GPU

# 提取 CIFAR-10 测试数据的特征
test_features, test_labels = extract_image_features(clip_model, clip_preprocess, cifar10_test_loader)

# 评估模型
def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
    return accuracy

# 比较原始特征和补全特征的模型性能
original_model = MLP(input_dim, hidden_dim, output_dim).to(device)
augmented_model = MLP(input_dim, hidden_dim, output_dim).to(device)

train_model(original_model, original_train_loader)
train_model(augmented_model, augmented_train_loader)

original_cifar10_accuracy = evaluate_model(original_model, test_features, test_labels)
augmented_cifar10_accuracy = evaluate_model(augmented_model, test_features, test_labels)

print(f"原始特征的 CIFAR-10 测试准确率: {original_cifar10_accuracy}")
print(f"补全特征的 CIFAR-10 测试准确率: {augmented_cifar10_accuracy}")