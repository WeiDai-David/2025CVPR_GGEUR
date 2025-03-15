import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, features, labels):
        # 确保 features 和 labels 是 numpy 数组或张量
        self.features = features if isinstance(features, torch.Tensor) else torch.from_numpy(features).float()
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义简单的分类模型
class MyNet(nn.Module):
    def __init__(self, num_classes=7):  # PACS 有 7 个类别
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)

# 训练函数 (局部客户端训练)
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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

# 测试函数 (全局模型评估)
def test(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    return accuracy

# FedOpt 聚合函数 (使用全局学习率 ηg 更新全局模型)
def fedopt_aggregation(global_model, client_models, global_optimizer):
    global_dict = global_model.state_dict()
    num_clients = len(client_models)

    # 平均更新来自客户端的参数
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(num_clients)], 0).mean(0)

    # 加载更新后的全局模型参数
    global_model.load_state_dict(global_dict)

    # 全局优化器更新全局模型 (使用 FedOpt 的优化器)
    global_optimizer.step()

    return global_model

# 加载每个客户端的原始特征和标签文件
def load_client_features(client_idx, dataset_name, base_dir):
    features, labels = [], []
    for class_idx in range(7):  # PACS 有 7 个类
        original_features_path = os.path.join(base_dir, dataset_name, f'client_{client_idx}_class_{class_idx}_original_features.npy')
        original_labels_path = os.path.join(base_dir, dataset_name, f'client_{client_idx}_class_{class_idx}_labels.npy')
        
        if os.path.exists(original_features_path) and os.path.exists(original_labels_path):
            class_features = np.load(original_features_path)
            class_labels = np.load(original_labels_path)
            features.append(class_features)
            labels.append(class_labels)
        else:
            print(f"客户端 {client_idx} 类别 {class_idx} 的特征或标签文件不存在")
    
    if features and labels:
        features = np.vstack(features)
        labels = np.concatenate(labels)
        return features, labels
    else:
        return None, None

# 加载测试集特征和标签
def load_test_features_labels(dataset_name, base_dir='./clip_pacs_test_features'):
    test_features_path = os.path.join(base_dir, dataset_name, f'{dataset_name}_test_features.npy')
    test_labels_path = os.path.join(base_dir, dataset_name, f'{dataset_name}_test_labels.npy')
    
    if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
        test_features = np.load(test_features_path)
        test_labels = np.load(test_labels_path)
        return torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)
    else:
        raise FileNotFoundError(f"{dataset_name} 的测试集特征或标签文件不存在")

# 联邦训练和评估函数 (FedOpt)
def federated_train_and_evaluate_fedopt(all_client_features, test_sets, communication_rounds, local_epochs, batch_size=16, learning_rate=0.001, global_learning_rate=0.5):
    global_model = MyNet(num_classes=7).to(device)  # PACS 有 7 个类
    client_models = [MyNet(num_classes=7).to(device) for _ in range(len(all_client_features))]

    criterion = nn.CrossEntropyLoss()

    # 定义全局优化器 (用于FedOpt的全局更新)
    global_optimizer = optim.SGD(global_model.parameters(), lr=global_learning_rate)

    report_path = './results/FedOpt_original_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    all_accuracies = {name: [] for name in list(test_sets.keys()) + ['average']}
    best_avg_accuracy = 0.0  # 用来追踪最高的平均精度
    best_model_state = None  # 用来保存精度最高时的模型状态

    with open(report_path, 'w') as f:
        for round in range(communication_rounds):
            # 客户端本地训练
            for client_idx, client_data in enumerate(all_client_features):
                original_features, original_labels = client_data
                client_model = client_models[client_idx]
                optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

                for epoch in range(local_epochs):
                    train_dataset = MyDataset(original_features, original_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    train_loss, train_accuracy = train(client_model, train_dataloader, criterion, optimizer, device)

            # FedOpt 聚合 (使用全局优化器更新全局模型)
            global_model = fedopt_aggregation(global_model, client_models, global_optimizer)

            # 在四个测试集上评估全局模型
            accuracies = []
            for dataset_name, (test_features, test_labels) in test_sets.items():
                test_dataset = MyDataset(test_features, test_labels)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                test_accuracy = test(global_model, test_dataloader, device)
                accuracies.append(test_accuracy)
                all_accuracies[dataset_name].append(test_accuracy)
            
            # 计算平均精度
            avg_accuracy = sum(accuracies) / len(accuracies)
            all_accuracies['average'].append(avg_accuracy)
            
            # 生成统一格式的输出字符串
            accuracy_output = (f"Round {round + 1}/{communication_rounds}, "
                               + ', '.join([f"{dataset_name.capitalize()} Accuracy: {test_accuracy:.4f}" for dataset_name, test_accuracy in zip(test_sets.keys(), accuracies)])
                               + f", Average Accuracy: {avg_accuracy:.4f}")
            
            # 输出到终端
            print(accuracy_output)
            
            # 保存到文件
            f.write(accuracy_output + "\n")

            
            # 检查是否是最高的平均精度
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_model_state = global_model.state_dict()  # 保存精度最高时的模型状态

    return best_model_state, all_accuracies

# 客户端编号分配
client_range = {
    'photo': [0],
    'art_painting': [1],
    'cartoon': [2],
    'sketch': [3]
}

# 绘制精度变化图
def plot_accuracies(accuracies, communication_rounds):
    plt.figure(figsize=(10, 6))
    rounds = range(1, communication_rounds + 1)
    
    for dataset_name, acc_values in accuracies.items():
        plt.plot(rounds, acc_values, label=f'{dataset_name} Accuracy')

    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy across Communication Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/FedOpt_original_test_accuracy_plot.png')
    plt.show()

# 主函数
def main():
    # 客户端数据加载路径
    base_dir = './clip_pacs_train_features'  # PACS 数据集的路径
    
    # 加载客户端的原始特征
    all_client_features = []
    datasets = ['photo', 'art_painting', 'cartoon', 'sketch']  # PACS 数据集
    for dataset_name in datasets:
        for client_id in client_range[dataset_name]:
            features, labels = load_client_features(client_id, dataset_name, base_dir)
            if features is not None and labels is not None:
                all_client_features.append((features, labels))

    # 加载测试集特征和标签
    test_sets = {}
    for dataset_name in datasets:
        test_features, test_labels = load_test_features_labels(dataset_name)
        test_sets[dataset_name] = (test_features, test_labels)
        
    os.makedirs('./model', exist_ok=True)  # 如果输出文件夹不存在则创建

    # 联邦训练和评估 (FedOpt)
    communication_rounds = 50
    local_epochs = 10
    global_learning_rate = 0.5  # 全局学习率

    best_model_state, all_accuracies = federated_train_and_evaluate_fedopt(all_client_features, test_sets, communication_rounds, local_epochs, learning_rate=0.001, global_learning_rate=global_learning_rate)

    # 保存精度最高的模型
    best_model_path = './model/FedOpt_original_best_model.pth'
    torch.save(best_model_state, best_model_path)

    # 绘制精度变化图
    plot_accuracies(all_accuracies, communication_rounds)

if __name__ == "__main__":
    main()
