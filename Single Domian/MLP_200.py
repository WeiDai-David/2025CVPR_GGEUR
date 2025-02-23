import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
import open_clip

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
    def __init__(self, num_classes=200):
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

# 加载客户端特征文件
def load_client_features(client_idx, base_dir, alpha):
    features_dict = {'original': {}, 'augmented': {}}
    for class_idx in range(200):
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

# 加载 TinyImageNet 的原始和补全特征文件
alpha = 0.2
base_dir = './TinyImageNet/features'
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
model_dir = f'./TinyImageNet/features/alpha={alpha}_model/client_{client_idx}'
os.makedirs(model_dir, exist_ok=True)

np.savetxt(os.path.join(model_dir, "original_features.txt"), original_features, delimiter=',')
np.savetxt(os.path.join(model_dir, "original_labels.txt"), original_labels, delimiter=',', fmt='%d')
np.savetxt(os.path.join(model_dir, "augmented_features.txt"), augmented_features, delimiter=',')
np.savetxt(os.path.join(model_dir, "augmented_labels.txt"), augmented_labels, delimiter=',', fmt='%d')

# 保存为 pkl 文件
save_as_pkl({'features': original_features, 'labels': original_labels}, os.path.join(model_dir, 'original_features.pkl'))
save_as_pkl({'features': augmented_features, 'labels': augmented_labels}, os.path.join(model_dir, 'augmented_features.pkl'))

# 加载 TinyImageNet 验证集特征和标签
val_features_dir = './TinyImageNet/val_features'
val_features = np.load(os.path.join(val_features_dir, 'val_final_embeddings.npy'))
val_labels = np.load(os.path.join(val_features_dir, 'val_labels.npy'))

# 转换为 torch.tensor
val_features = torch.tensor(val_features, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.long)

print(f"验证集特征大小: {val_features.shape}")
print(f"验证集标签大小: {val_labels.shape}")

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

    model = MyNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0.0
    best_epoch = 0
    epoch_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        test_accuracy, _ = evaluate_model(model, test_features, test_labels)
        epoch_accuracies.append(test_accuracy * 100)  # 将准确率转换为百分比

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

    print(f"Best {model_name} Test Accuracy: {best_test_accuracy:.4f} at Epoch {best_epoch}")

    # 加载最佳模型并在验证集上评估
    model.load_state_dict(torch.load(model_path))
    test_accuracy, report = evaluate_model(model, test_features, test_labels)
    output_content = (
        f"{model_name} 的 TinyImageNet 测试准确率: {test_accuracy:.4f}\n"
        f"{model_name} 的 TinyImageNet 分类报告:\n{report}\n"
    )
    print(output_content)

    # 将输出内容保存到文件
    with open(output_file_path, 'a') as f:
        f.write(output_content)

    return epoch_accuracies

# 设置训练参数
batch_size = 64
learning_rate = 0.001
num_epochs = 40

# 训练和评估原始特征模型
original_model_path = os.path.join(model_dir, 'original_tinyimagenet_best_model.pth')
original_epoch_accuracies = train_and_evaluate({'features': original_features, 'labels': original_labels}, original_model_path, "原始特征模型", val_features, val_labels, f'./TinyImageNet/features/alpha={alpha}_model/client_{client_idx}/original_report.txt')

# 训练和评估补全特征模型
augmented_model_path = os.path.join(model_dir, 'augmented_tinyimagenet_best_model.pth')
augmented_epoch_accuracies = train_and_evaluate({'features': augmented_features, 'labels': augmented_labels}, augmented_model_path, "补全特征模型", val_features, val_labels, f'./TinyImageNet/features/alpha={alpha}_model/client_{client_idx}/augmented_report.txt')

# 绘制对比图，保持现有的区间
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, original_epoch_accuracies, 'r-', label='Original feature model')
plt.plot(epochs, augmented_epoch_accuracies, 'b-', label='Augment feature model')
plt.xlabel('Epoch')
plt.ylabel('top-1 accuracy (%)')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())  # 将 y 轴标签改为百分比
plt.title(f'alpha={alpha}_TinyImageNet_client_idx={client_idx}')
plt.legend()
plt.grid(True)
plt.savefig(f'./TinyImageNet/features/alpha={alpha}_model/client_{client_idx}/accuracy_comparison.png')
plt.show()

# 绘制对比图，设置纵坐标区间为 0-100
plt.figure(figsize=(10, 6))
plt.plot(epochs, original_epoch_accuracies, 'r-', label='Original feature model')
plt.plot(epochs, augmented_epoch_accuracies, 'b-', label='Augment feature model')
plt.xlabel('Epoch')
plt.ylabel('top-1 accuracy (%)')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())  # 将 y 轴标签改为百分比
plt.ylim(0, 100)  # 设置 y 轴区间为 0-100
plt.title(f'alpha={alpha}_TinyImageNet_client_idx={client_idx}')
plt.legend()
plt.grid(True)
plt.savefig(f'./TinyImageNet/features/alpha={alpha}_model/client_{client_idx}/accuracy_comparison_0-100_scale.png')
plt.show()
