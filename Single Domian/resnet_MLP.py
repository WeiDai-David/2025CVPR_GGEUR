import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms, models
from PIL import Image
import warnings
import matplotlib.pyplot as plt

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
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc3(x)
        return F.softmax(x, dim=1)

# 定义自定义的 ResNet50 模型
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

# 加载客户端特征文件
def load_client_features(client_idx, base_dir, alpha):
    features_dict = {'original': {}, 'augmented': {}}
    for class_idx in range(10):
        original_features_path = os.path.join(base_dir, 'resnet_initial', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'final_embeddings.npy')
        augmented_features_path = os.path.join(base_dir, f'resnet_alpha={alpha}_complete', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'final_embeddings_filled.npy')
        original_labels_path = os.path.join(base_dir, 'resnet_initial', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'labels.npy')
        augmented_labels_path = os.path.join(base_dir, f'resnet_alpha={alpha}_complete', f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')

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

# 定义 unpickle 函数
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 定义训练函数
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

# 加载保存的测试特征和标签
def load_saved_test_features_and_labels(base_dir):
    features_path = os.path.join(base_dir, 'final_embeddings.npy')
    labels_path = os.path.join(base_dir, 'labels.npy')
    if os.path.exists(features_path) and os.path.exists(labels_path):
        features = np.load(features_path)
        labels = np.load(labels_path)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    return None, None

# 分别训练和评估原始特征模型和补全特征模型
def train_and_evaluate(train_dict, model_path, model_name, test_features, test_labels, output_file_path):
    train_dataset = MyDataset(train_dict['features'], train_dict['labels'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MyNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0.0
    best_epoch = 0
    test_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        test_accuracy, _ = evaluate_model(model, test_features, test_labels)

        test_accuracies.append(test_accuracy * 100)  # 保存测试精度（百分比）

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # 保留测试精度最高的模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

    print(f"Best {model_name} Test Accuracy: {best_test_accuracy:.4f} at Epoch {best_epoch}")

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(model_path))
    test_accuracy, report = evaluate_model(model, test_features, test_labels)
    output_content = (
        f"{model_name} 的 CIFAR-10 测试准确率: {test_accuracy:.4f}\n"
        f"{model_name} 的 CIFAR-10 分类报告:\n{report}\n"
        f"Best {model_name} Test Accuracy: {best_test_accuracy:.4f} at Epoch {best_epoch}\n"
    )
    print(output_content)

    # 将输出内容保存到文件
    with open(output_file_path, 'a') as f:
        f.write(output_content)

    return test_accuracies

# 画精度变化图
def plot_accuracies(test_accuracies, model_name, output_dir):
    epochs = list(range(1, len(test_accuracies) + 1))
    
    # 默认区间的图
    plt.figure()
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Test Accuracy over Epochs (Default Range)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_test_accuracy_default_range.png'))
    plt.close()

    # 0%-100% 区间的图
    plt.figure()
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} Test Accuracy over Epochs (0%-100%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_test_accuracy_0_to_100.png'))
    plt.close()

# 设置训练参数
batch_size = 64
learning_rate = 0.001
num_epochs = 60

# 加载特征文件并合并
alpha = 0.1
base_dir = './CIFAR-10/features'
client_idx = 9

# 加载客户端的原始和补全特征文件
features_dict = load_client_features(client_idx, base_dir, alpha)
original_features, original_labels = merge_features(features_dict['original'])
augmented_features, augmented_labels = merge_features(features_dict['augmented'])

# 加载 CIFAR-10 测试数据并提取特征
test_features_dir = f'./CIFAR-10/features/resnet_test'
test_features, test_labels = load_saved_test_features_and_labels(test_features_dir)

# 提取测试集特征并保存
if test_features is None or test_labels is None:
    print("create test_features...create test_labels...")
    
    # 使用自定义的 ResNet50 模型提取特征
    resnet_model = CustomResNet().to(device)
    resnet_model.eval()

    # CIFAR-10 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试数据
    test_data_batch_file = './data/cifar-10-batches-py/test_batch'
    test_data_batch = unpickle(test_data_batch_file)
    test_data = test_data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
    test_labels = np.array(test_data_batch[b'labels'])

    test_features_list = []
    with torch.no_grad():
        for img_array in test_data:
            # 检查图像数据类型，并进行必要的转换
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img = preprocess(img).unsqueeze(0).to(device)
            features = resnet_model(img)
            test_features_list.append(features.cpu())

    test_features = torch.cat(test_features_list, dim=0)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

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

# 设置模型保存路径
model_dir = f'./CIFAR-10/features/resnet_alpha={alpha}_model/client_{client_idx}'
os.makedirs(model_dir, exist_ok=True)

# 训练和评估原始特征模型
original_model_path = os.path.join(model_dir, 'original_cifar10_best_model.pth')
test_accuracies = train_and_evaluate({'features': original_features, 'labels': original_labels}, original_model_path, "原始特征模型", test_features, test_labels, os.path.join(model_dir, 'original_report.txt'))

# 绘制原始特征模型的精度变化图
plot_accuracies(test_accuracies, "original_model", model_dir)

# 训练和评估补全特征模型
augmented_model_path = os.path.join(model_dir, 'augmented_cifar10_best_model.pth')
test_accuracies = train_and_evaluate({'features': augmented_features, 'labels': augmented_labels}, augmented_model_path, "补全特征模型", test_features, test_labels, os.path.join(model_dir, 'augmented_report.txt'))

# 绘制补全特征模型的精度变化图
plot_accuracies(test_accuracies, "augmented_model", model_dir)
