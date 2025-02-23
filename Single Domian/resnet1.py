# 使用提取的原始特征训练原始特征模型
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

# 自定义数据集类，用于加载特征和标签
class FeatureDataset(Dataset):
    def __init__(self, feature_dir, client_id=None, is_test=False):
        self.features = []
        self.labels = []
        for class_idx in range(10):
            if is_test:
                feature_path = os.path.join(feature_dir, f'test_class_{class_idx}_features.npy')
                label_path = os.path.join(feature_dir, f'test_class_{class_idx}_labels.npy')
            else:
                feature_path = os.path.join(feature_dir, f'alpha=0.1_class_{class_idx}_client_{client_id}/final_embeddings.npy')
                label_path = os.path.join(feature_dir, f'alpha=0.1_class_{class_idx}_client_{client_id}/labels.npy')
                
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
    model = resnet50(weights=None)  # 不加载预训练权重
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)  # 将全连接层设置为10分类
    return model

def train_and_evaluate(client_id, feature_dir, test_feature_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载训练特征数据
    train_dataset = FeatureDataset(feature_dir, client_id=client_id)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 加载测试特征数据
    test_dataset = FeatureDataset(test_feature_dir, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 加载ResNet50模型并替换全连接层
    model = load_resnet_model().to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 训练模型的全连接层
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

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

    # 保存重新训练的模型到initialmodel目录
    os.makedirs(f'./CIFAR-10/features/resnet/initialmodel', exist_ok=True)
    torch.save(model.state_dict(), f'./CIFAR-10/features/resnet/initialmodel/resnet_cifar10_client_{client_id}.pth')

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
    feature_dir = './CIFAR-10/features/resnet/initial'
    test_feature_dir = './CIFAR-10/features/resnet/test'

    # 处理所有客户端的数据，并对每个客户端进行评估
    for client_id in range(10):
        train_and_evaluate(client_id, feature_dir, test_feature_dir)

if __name__ == "__main__":
    main()
