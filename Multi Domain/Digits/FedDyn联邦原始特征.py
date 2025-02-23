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
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义简单的分类模型
class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)

# FedDyn 聚合函数
def feddyn_update(global_model, client_models, client_grads, lambda_dyn):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        for i, client_model in enumerate(client_models):
            # 更新模型权重，考虑梯度调整
            global_dict[k] += lambda_dyn * client_grads[i][k]
        # 将更新后的权重加载回全局模型
        global_dict[k] = global_dict[k] / len(client_models)
    global_model.load_state_dict(global_dict)
    return global_model

# 初始化客户端梯度字典
def initialize_client_grads(client_models):
    client_grads = {}
    for i, client_model in enumerate(client_models):
        client_grads[i] = {}
        for name, param in client_model.named_parameters():
            client_grads[i][name] = torch.zeros_like(param)
    return client_grads

# 修改训练函数，加入 Proximal Weight α
def train_feddyn(model, global_model, dataloader, criterion, optimizer, device, client_grads, lambda_dyn, alpha=0.1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    global_model_params = dict(global_model.named_parameters())
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 加入 Proximal Regularization (基于全局模型)
        prox_loss = 0.0
        for name, param in model.named_parameters():
            prox_loss += (alpha / 2.0) * torch.norm(param - global_model_params[name].detach()) ** 2
        
        # 总损失 = 原损失 + Proximal 损失
        total_loss = loss + prox_loss

        total_loss.backward()

        # 更新梯度字典
        for name, param in model.named_parameters():
            client_grads[name].add_(param.grad)

        optimizer.step()

        running_loss += total_loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total if total > 0 else 0
    return epoch_loss, epoch_accuracy, client_grads

# 测试函数
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

# 解析 dataset_report.txt 文件，确定每个数据集对应的客户端
def parse_dataset_report(report_file):
    dataset_clients = {}
    with open(report_file, 'r') as f:
        lines = f.readlines()
        current_dataset = None
        for line in lines:
            if "数据集大小" in line:
                current_dataset = line.split()[0]
                dataset_clients[current_dataset] = []
            elif "客户端" in line:
                client_id = int(line.split()[1])
                dataset_clients[current_dataset].append(client_id)
    return dataset_clients

# 加载训练集特征
def load_training_data(dataset_name, client_id, base_dir):
    all_features = []
    all_labels = []

    for class_idx in range(10):
        feature_file = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_idx}_original_features.npy')
        label_file = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_idx}_labels.npy')

        if os.path.exists(feature_file) and os.path.exists(label_file):
            print(f"加载 {dataset_name} 客户端 {client_id} 类别 {class_idx} 的特征和标签文件")
            features = np.load(feature_file)
            labels = np.load(label_file)
            all_features.append(features)
            all_labels.append(labels)
        else:
            raise FileNotFoundError(f"客户端 {client_id} 类别 {class_idx} 的特征或标签文件不存在")

    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return torch.tensor(all_features, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.long)

# 加载测试集特征
def load_test_data(dataset_name, base_dir):
    feature_file = os.path.join(base_dir, f'{dataset_name}/{dataset_name}_test_features.npy')
    label_file = os.path.join(base_dir, f'{dataset_name}/{dataset_name}_test_labels.npy')

    if os.path.exists(feature_file) and os.path.exists(label_file):
        print(f"加载 {dataset_name} 测试集的特征和标签文件")
        features = np.load(feature_file)
        labels = np.load(label_file)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    else:
        raise FileNotFoundError(f"测试集 {dataset_name} 的特征或标签文件不存在")

# 联邦训练和评估函数 (FedDyn 版本)
def federated_train_and_evaluate_feddyn(all_client_features, test_loaders, communication_rounds, local_epochs, batch_size=16, learning_rate=0.01, lambda_dyn=1.0, alpha=0.1):
    global_model = MyNet(num_classes=10).to(device)
    client_models = [MyNet(num_classes=10).to(device) for _ in range(len(all_client_features))]
    client_grads = initialize_client_grads(client_models)

    criterion = nn.CrossEntropyLoss()

    test_accuracies_mnist = []
    test_accuracies_usps = []
    test_accuracies_svhn = []
    test_accuracies_syn = []
    avg_accuracies = []

    best_avg_accuracy = 0.0  # 记录最高的平均精度
    best_model_state = None  # 保存精度最高时的模型状态

    report_path = './results/FedDyn_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        for round in range(communication_rounds):
            # 客户端训练
            for client_idx, client_data in enumerate(all_client_features):
                original_features, original_labels = client_data
                client_model = client_models[client_idx]
                optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

                for epoch in range(local_epochs):
                    train_dataset = MyDataset(original_features, original_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    train_loss, train_accuracy, client_grads[client_idx] = train_feddyn(
                        client_model, global_model, train_dataloader, criterion, optimizer, device, client_grads[client_idx], lambda_dyn, alpha)

            # FedDyn 更新
            global_model = feddyn_update(global_model, client_models, client_grads, lambda_dyn)

            # 在四个测试集上分别评估全局模型
            mnist_features, mnist_labels = test_loaders['mnist']
            usps_features, usps_labels = test_loaders['usps']
            svhn_features, svhn_labels = test_loaders['svhn']
            syn_features, syn_labels = test_loaders['syn']

            test_accuracy_mnist = test(global_model, DataLoader(MyDataset(mnist_features, mnist_labels), batch_size=batch_size), device)
            test_accuracy_usps = test(global_model, DataLoader(MyDataset(usps_features, usps_labels), batch_size=batch_size), device)
            test_accuracy_svhn = test(global_model, DataLoader(MyDataset(svhn_features, svhn_labels), batch_size=batch_size), device)
            test_accuracy_syn = test(global_model, DataLoader(MyDataset(syn_features, syn_labels), batch_size=batch_size), device)

            avg_accuracy = (test_accuracy_mnist + test_accuracy_usps + test_accuracy_svhn + test_accuracy_syn) / 4

            test_accuracies_mnist.append(test_accuracy_mnist * 100)
            test_accuracies_usps.append(test_accuracy_usps * 100)
            test_accuracies_svhn.append(test_accuracy_svhn * 100)
            test_accuracies_syn.append(test_accuracy_syn * 100)
            avg_accuracies.append(avg_accuracy * 100)

            print(f"Communication Round {round + 1}/{communication_rounds}, MNIST Accuracy: {test_accuracy_mnist:.4f}, USPS Accuracy: {test_accuracy_usps:.4f}, SVHN Accuracy: {test_accuracy_svhn:.4f}, SYN Accuracy: {test_accuracy_syn:.4f}, Average Accuracy: {avg_accuracy:.4f}")
            f.write(f"Communication Round {round + 1}/{communication_rounds}, MNIST Accuracy: {test_accuracy_mnist:.4f}, USPS Accuracy: {test_accuracy_usps:.4f}, SVHN Accuracy: {test_accuracy_svhn:.4f}, SYN Accuracy: {test_accuracy_syn:.4f}, Average Accuracy: {avg_accuracy:.4f}\n")

            # 如果当前的平均精度超过之前的最佳精度，则保存当前模型
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_model_state = global_model.state_dict()  # 保存最佳模型的状态

    # 返回最佳模型状态
    return best_model_state, test_accuracies_mnist, test_accuracies_usps, test_accuracies_svhn, test_accuracies_syn, avg_accuracies

# 主函数
def main():
    # 加载客户端-数据集映射
    report_file = './output_indices/dataset_report.txt'
    dataset_clients = parse_dataset_report(report_file)

    # 加载训练集客户端的特征和标签
    train_base_dir = './clip_features'
    all_client_features = []
    for dataset_name, clients in dataset_clients.items():
        for client_id in clients:
            features, labels = load_training_data(dataset_name, client_id, train_base_dir)
            all_client_features.append((features, labels))

    # 加载测试集特征和标签
    test_base_dir = './clip_test_features'
    test_loaders = {
        'mnist': load_test_data('mnist', test_base_dir),  # 测试集没有按客户端划分
        'usps': load_test_data('usps', test_base_dir),
        'svhn': load_test_data('svhn', test_base_dir),
        'syn': load_test_data('syn', test_base_dir)
    }

    # 联邦训练和评估 (FedDyn)
    communication_rounds = 100
    local_epochs = 10
    best_model_state, test_accuracies_mnist, test_accuracies_usps, test_accuracies_svhn, test_accuracies_syn, avg_accuracies = federated_train_and_evaluate_feddyn(
        all_client_features, test_loaders, communication_rounds, local_epochs)

    # 保存精度最高的模型
    best_model_path = './model/FedDyn_global_model.pth'
    torch.save(best_model_state, best_model_path)

    # 绘制测试集精度随通信轮次变化的折线图
    plt.figure()
    plt.plot(range(1, communication_rounds + 1), test_accuracies_mnist, label='MNIST Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_usps, label='USPS Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_svhn, label='SVHN Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_syn, label='SYN Accuracy')
    plt.plot(range(1, communication_rounds + 1), avg_accuracies, label='Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(0, 100)
    plt.title('Test Accuracy vs Communication Rounds')
    plt.legend()
    plt.savefig('./results/FedDyn_test_accuracy_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
