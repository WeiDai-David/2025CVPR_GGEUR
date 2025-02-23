import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

# 自定义数据集类，用于加载补全的特征和标签
class FeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.features = []
        self.labels = []
        for class_idx in range(10):
            for client_idx in range(10):
                feature_path = os.path.join(feature_dir, f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'final_embeddings_filled.npy')
                label_path = os.path.join(feature_dir, f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')
                if os.path.exists(feature_path) and os.path.exists(label_path):
                    features = np.load(feature_path)
                    labels = np.load(label_path)
                    self.features.append(features)
                    self.labels.append(labels)
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 自定义数据集类，用于加载测试集的特征和标签
class TestFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.features = []
        self.labels = []
        for class_idx in range(10):
            feature_path = os.path.join(feature_dir, f'test_class_{class_idx}_features.npy')
            label_path = os.path.join(feature_dir, f'test_class_{class_idx}_labels.npy')
            if os.path.exists(feature_path) and os.path.exists(label_path):
                features = np.load(feature_path)
                labels = np.load(label_path)
                self.features.append(features)
                self.labels.append(labels)
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_resnet_model():
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    return model

def train_and_evaluate(client_id, feature_dir, test_feature_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载补全特征数据
    train_dataset = FeatureDataset(feature_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 加载测试集特征数据
    test_dataset = TestFeatureDataset(test_feature_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 加载原始特征训练的ResNet50模型
    model = load_resnet_model().to(device)
    model.load_state_dict(torch.load(f'./CIFAR-10/features/resnet/initialmodel/resnet_cifar10_client_{client_id}.pth'))

    # 冻结除全连接层外的所有层
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 继续训练模型
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

            # 启用梯度计算
            inputs.requires_grad_()

            optimizer.zero_grad()
            outputs = model.fc(inputs)  # 仅训练全连接层
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Client {client_id}, Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print(f'Finished Training Client {client_id}')

    # 保存重新训练的模型
    os.makedirs(f'./CIFAR-10/features/resnet/completemodel', exist_ok=True)
    torch.save(model.state_dict(), f'./CIFAR-10/features/resnet/completemodel/resnet_cifar10_client_{client_id}.pth')

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model.fc(inputs)  # 仅测试全连接层
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images for client {client_id}: {100 * correct / total:.2f}%')

def main():
    feature_dir = './CIFAR-10/features/resnet/complete'
    test_feature_dir = './CIFAR-10/features/resnet/test'

    # 处理所有客户端的数据，并对每个客户端进行评估
    for client_id in range(10):
        train_and_evaluate(client_id, feature_dir, test_feature_dir)

if __name__ == "__main__":
    main()
