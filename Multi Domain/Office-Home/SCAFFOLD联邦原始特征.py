import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

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
    def __init__(self, num_classes=65):  # 修改为65个类别
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)

# 初始化控制变量
def init_control(model):
    control = {name: torch.zeros_like(param) for name, param in model.state_dict().items()}
    return control

# SCAFFOLD 训练函数
def train_scaffold(model, global_control, local_control, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 控制变量校正
        correction_term = 0
        for name, param in model.named_parameters():
            correction_term += torch.sum((global_control[name] - local_control[name]) * param)

        loss += correction_term
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct / total if total > 0 else 0
    return epoch_loss, epoch_accuracy

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

# SCAFFOLD 聚合函数
def scaffold_aggregation(global_model, client_models, global_control, local_controls, delta_models):
    global_dict = global_model.state_dict()
    num_clients = len(client_models)

    # 更新全局模型参数
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(num_clients)], 0).mean(0)
    
    # 更新全局控制变量
    for name in global_control.keys():
        control_delta_sum = torch.stack([delta_models[i][name] for i in range(num_clients)], 0).mean(0)
        global_control[name] -= control_delta_sum

    global_model.load_state_dict(global_dict)
    return global_model, global_control

# 加载训练集特征
def load_training_data(dataset_name, client_id, base_dir):
    all_features = []
    all_labels = []

    for class_idx in range(65):  # 修改为65个类别
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

# 联邦训练和评估函数 (SCAFFOLD)
def federated_train_and_evaluate_scaffold(all_client_features, test_loaders, communication_rounds, local_epochs, batch_size=16, learning_rate=0.01):
    global_model = MyNet(num_classes=65).to(device)  # 修改为65个类别
    client_models = [MyNet(num_classes=65).to(device) for _ in range(len(all_client_features))]

    # 初始化控制变量
    global_control = init_control(global_model)
    local_controls = [init_control(client_models[i]) for i in range(len(client_models))]
    delta_models = [{} for _ in range(len(client_models))]

    criterion = nn.CrossEntropyLoss()

    test_accuracies_art = []
    test_accuracies_clipart = []
    test_accuracies_product = []
    test_accuracies_realworld = []
    avg_accuracies = []

    best_avg_accuracy = 0.0  # 记录最高的平均精度
    best_model_state = None  # 保存精度最高时的模型状态

    report_path = './results/SCAFFOLD_office_home_report.txt'
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
                    train_loss, train_accuracy = train_scaffold(client_model, global_control, local_controls[client_idx], train_dataloader, criterion, optimizer, device)

                # 计算模型更新的变化量
                delta_models[client_idx] = {
                    name: (global_control[name] - local_controls[client_idx][name]) for name in global_control.keys()
                }

            # SCAFFOLD 聚合
            global_model, global_control = scaffold_aggregation(global_model, client_models, global_control, local_controls, delta_models)

            # 在四个测试集上分别评估全局模型
            art_features, art_labels = test_loaders['Art']
            clipart_features, clipart_labels = test_loaders['Clipart']
            product_features, product_labels = test_loaders['Product']
            realworld_features, realworld_labels = test_loaders['Real_World']

            test_accuracy_art = test(global_model, DataLoader(MyDataset(art_features, art_labels), batch_size=batch_size), device)
            test_accuracy_clipart = test(global_model, DataLoader(MyDataset(clipart_features, clipart_labels), batch_size=batch_size), device)
            test_accuracy_product = test(global_model, DataLoader(MyDataset(product_features, product_labels), batch_size=batch_size), device)
            test_accuracy_realworld = test(global_model, DataLoader(MyDataset(realworld_features, realworld_labels), batch_size=batch_size), device)

            avg_accuracy = (test_accuracy_art + test_accuracy_clipart + test_accuracy_product + test_accuracy_realworld) / 4

            test_accuracies_art.append(test_accuracy_art * 100)
            test_accuracies_clipart.append(test_accuracy_clipart * 100)
            test_accuracies_product.append(test_accuracy_product * 100)
            test_accuracies_realworld.append(test_accuracy_realworld * 100)
            avg_accuracies.append(avg_accuracy * 100)

            print(f"Communication Round {round + 1}/{communication_rounds}, Art Accuracy: {test_accuracy_art:.4f}, Clipart Accuracy: {test_accuracy_clipart:.4f}, Product Accuracy: {test_accuracy_product:.4f}, Real-World Accuracy: {test_accuracy_realworld:.4f}, Average Accuracy: {avg_accuracy:.4f}")
            f.write(f"Communication Round {round + 1}/{communication_rounds}, Art Accuracy: {test_accuracy_art:.4f}, Clipart Accuracy: {test_accuracy_clipart:.4f}, Product Accuracy: {test_accuracy_product:.4f}, Real-World Accuracy: {test_accuracy_realworld:.4f}, Average Accuracy: {avg_accuracy:.4f}\n")

            # 如果当前的平均精度超过之前的最佳精度，则保存当前模型
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_model_state = global_model.state_dict()  # 保存最佳模型的状态

    # 返回最佳模型状态
    return best_model_state, test_accuracies_art, test_accuracies_clipart, test_accuracies_product, test_accuracies_realworld, avg_accuracies

# 主函数
def main():
    # 加载客户端-数据集映射
    client_range = {
        'Art': [0],
        'Clipart': [1],
        'Product': [2],
        'Real_World': [3]
    }

    # 加载训练集客户端的特征和标签
    train_base_dir = './clip_office_home_train_features'
    all_client_features = []
    for dataset_name, clients in client_range.items():
        for client_id in clients:
            features, labels = load_training_data(dataset_name, client_id, train_base_dir)
            all_client_features.append((features, labels))

    # 加载测试集特征和标签
    test_base_dir = './clip_office_home_test_features'
    test_loaders = {
        'Art': load_test_data('Art', test_base_dir),
        'Clipart': load_test_data('Clipart', test_base_dir),
        'Product': load_test_data('Product', test_base_dir),
        'Real_World': load_test_data('Real_World', test_base_dir)
    }

    # 联邦训练和评估
    communication_rounds = 50
    local_epochs = 10
    best_model_state, test_accuracies_art, test_accuracies_clipart, test_accuracies_product, test_accuracies_realworld, avg_accuracies = federated_train_and_evaluate_scaffold(
        all_client_features, test_loaders, communication_rounds, local_epochs)

    # 保存精度最高的模型
    best_model_path = './model/SCAFFOLD_office_home_global_model.pth'
    torch.save(best_model_state, best_model_path)

    # 绘制测试集精度随通信轮次变化的折线图
    plt.figure()
    plt.plot(range(1, communication_rounds + 1), test_accuracies_art, label='Art Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_clipart, label='Clipart Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_product, label='Product Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_realworld, label='Real-World Accuracy')
    plt.plot(range(1, communication_rounds + 1), avg_accuracies, label='Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(0, 100)
    plt.title('Test Accuracy vs Communication Rounds (SCAFFOLD)')
    plt.legend()
    plt.savefig('./results/SCAFFOLD_office_home_test_accuracy_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
