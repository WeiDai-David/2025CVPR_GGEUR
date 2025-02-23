import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import warnings
import open_clip
import sys
import contextlib


warnings.filterwarnings("ignore", category=UserWarning, message="Torch was not compiled with flash attention")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集类，用于加载特征和标签
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]).float(), torch.tensor(self.labels[idx], dtype=torch.long))

class MyNet(nn.Module):
    def __init__(self, num_classes=100):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc3(x)
        return F.softmax(x, dim=1)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total if total > 0 else 0

    return epoch_loss, epoch_accuracy

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total if total > 0 else 0

    return accuracy, all_labels, all_preds

# 定义特征提取函数
def extract_image_features(model, data, indices, preprocess):
    features_list = []
    with torch.no_grad():
        for index in indices:
            img_array = data[index]
            img = Image.fromarray(img_array)
            img = preprocess(img).unsqueeze(0).to(device)
            features = model.encode_image(img)
            features_list.append(features.cpu())
    return torch.cat(features_list)

# 解析 CIFAR-100 数据集的批次文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载客户端特征文件
def load_client_features(client_idx, base_dir, alpha):
    features_dict = {'original': {}, 'augmented': {}}
    for class_idx in range(100):
        original_features_path = os.path.join(base_dir, 'initial', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'final_embeddings.npy')
        augmented_features_path = os.path.join(base_dir, f'alpha={alpha}_complete', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'final_embeddings_filled.npy')
        original_labels_path = os.path.join(base_dir, 'initial', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'labels.npy')
        augmented_labels_path = os.path.join(base_dir, f'alpha={alpha}_complete', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')

        if os.path.exists(original_features_path):
            original_features = np.load(original_features_path)
            original_labels = np.load(original_labels_path)
            features_dict['original'][class_idx] = (original_features, original_labels)

        if os.path.exists(augmented_features_path):
            augmented_features = np.load(augmented_features_path)
            augmented_labels = np.load(augmented_labels_path)
            features_dict['augmented'][class_idx] = (augmented_features, augmented_labels)

    return features_dict

# 合并特征和标签
def merge_features(features_dict):
    merged_features = []
    merged_labels = []
    for class_idx in features_dict:
        features, labels = features_dict[class_idx]
        merged_features.append(features)
        merged_labels.append(labels)
    merged_features = np.vstack(merged_features)
    merged_labels = np.concatenate(merged_labels)
    return merged_features, merged_labels

# 加载特征文件并合并
alpha = 0.2
base_dir = './CIFAR-100/features'
client_idx = 9

# 加载客户端的原始和补全特征文件
features_dict = load_client_features(client_idx, base_dir, alpha)

# 合并原始特征和补全特征
original_features, original_labels = merge_features(features_dict['original'])
augmented_features, augmented_labels = merge_features(features_dict['augmented'])

# 打印特征和标签的大小和内容示例
print("原始特征大小:", original_features.shape)
print("原始标签大小:", original_labels.shape)
print("原始特征示例:\n", original_features)
print("原始标签示例:\n", original_labels)
print("补全特征大小:", augmented_features.shape)
print("补全标签大小:", augmented_labels.shape)
print("补全特征示例:\n", augmented_features)
print("补全标签示例:\n", augmented_labels)

# 保存为 pkl 文件
def save_as_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# 保存特征和标签到文件
model_dir = f'./CIFAR-100/features/alpha={alpha}_model/client_{client_idx}'
os.makedirs(model_dir, exist_ok=True)

np.savetxt(os.path.join(model_dir, "original_features.txt"), original_features, delimiter=',')
np.savetxt(os.path.join(model_dir, "original_labels.txt"), original_labels, delimiter=',', fmt='%d')
np.savetxt(os.path.join(model_dir, "augmented_features.txt"), augmented_features, delimiter=',')
np.savetxt(os.path.join(model_dir, "augmented_labels.txt"), augmented_labels, delimiter=',', fmt='%d')

# 保存为 pkl 文件
save_as_pkl({'features': original_features, 'labels': original_labels}, os.path.join(model_dir, 'original_features.pkl'))
save_as_pkl({'features': augmented_features, 'labels': augmented_labels}, os.path.join(model_dir, 'augmented_features.pkl'))

# 加载预训练的 CLIP 模型
backbone = 'ViT-B-32'
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'
clip_model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
clip_model.eval().to(device)

# 加载 CIFAR-100 测试数据并提取特征
test_features_dir = f'./CIFAR-100/features/alpha={alpha}_test'
test_features, test_labels = None, None

def load_saved_test_features_and_labels(base_dir):
    features_path = os.path.join(base_dir, 'final_embeddings.npy')
    labels_path = os.path.join(base_dir, 'labels.npy')
    if os.path.exists(features_path) and os.path.exists(labels_path):
        features = np.load(features_path)
        labels = np.load(labels_path)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    return None, None

# 检查是否存在已保存的测试特征和标签
test_features, test_labels = load_saved_test_features_and_labels(test_features_dir)

if test_features is None or test_labels is None:
    # 加载测试数据
    print("create test_features...create test_labels...")
    test_data_batch_file = './data/cifar-100-python/test'
    test_data_batch = unpickle(test_data_batch_file)
    test_data = test_data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
    test_labels = np.array(test_data_batch[b'fine_labels'])

    test_indices = np.arange(len(test_data))
    test_features = extract_image_features(clip_model, test_data, test_indices, preprocess)
    test_labels = torch.tensor(test_labels)

    # 保存测试特征和标签
    os.makedirs(test_features_dir, exist_ok=True)
    np.save(os.path.join(test_features_dir, 'final_embeddings.npy'), test_features.numpy())
    np.save(os.path.join(test_features_dir, 'labels.npy'), test_labels.numpy())
    print("提取并保存测试特征和标签")
else:
    print("使用已保存的测试特征和标签")
print(f"测试集图像数量: {test_features.shape[0]}")
print(f"测试集标签数量: {test_labels.shape[0]}")
print(f"提取的测试集特征数量: {test_features.shape[0]}")
print(f"测试集标签数量: {test_labels.shape[0]}")

# 评估模型
def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        report = classification_report(labels.cpu().numpy(), predicted.cpu().numpy(), digits=4)
    return accuracy, report

# 分别训练和评估原始特征模型和补全特征模型
def train_and_evaluate(train_dict, model_path, model_name, test_features, test_labels, output_file_path):
    train_dataset = MyDataset(train_dict['features'], train_dict['labels'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MyNet(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        torch.save(model.state_dict(), model_path)
        best_test_accuracy = train_accuracy
        best_epoch = epoch + 1

    print(f"Best {model_name} Train Accuracy: {best_test_accuracy:.4f} at Epoch {best_epoch}")

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(model_path))
    test_accuracy, report = evaluate_model(model, test_features, test_labels)
    output_content = (
        f"{model_name} 的 CIFAR-100 测试准确率: {test_accuracy:.4f}\n"
        f"{model_name} 的 CIFAR-100 分类报告:\n{report}\n"
    )
    print(output_content)

    # 将输出内容保存到文件
    with open(output_file_path, 'a') as f:
        f.write(output_content)


# 设置训练参数
batch_size = 64
learning_rate = 0.001
num_epochs = 60

# 训练和评估原始特征模型
original_model_path = os.path.join(model_dir, 'original_cifar100_best_model.pth')
train_and_evaluate({'features': original_features, 'labels': original_labels}, original_model_path, "原始特征模型", test_features, test_labels, f'./CIFAR-100/features/alpha={alpha}_model/client_{client_idx}/original_report.txt')

# 训练和评估补全特征模型
augmented_model_path = os.path.join(model_dir, 'augmented_cifar100_best_model.pth')
train_and_evaluate({'features': augmented_features, 'labels': augmented_labels}, augmented_model_path, "补全特征模型", test_features, test_labels, f'./CIFAR-100/features/alpha={alpha}_model/client_{client_idx}/augmented_report.txt')
