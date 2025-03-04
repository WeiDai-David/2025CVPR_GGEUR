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
        self.features = features if isinstance(features, torch.Tensor) else torch.from_numpy(features).float()
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.from_numpy(labels).long()

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

# 训练函数 (包含FedDyn的近端项)
def train_feddyn(model, global_model, delta_c, dataloader, criterion, optimizer, device, alpha=0.5):
    model.train()
    global_model.eval()
    running_loss, correct, total = 0.0, 0, 0

    global_state_dict = global_model.state_dict()  # 获取全局模型的状态字典

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 获取模型输出
        outputs = model(inputs)

        # 计算分类损失
        loss = criterion(outputs, labels)

        # 计算FedDyn校正项 (控制变量的调整)
        feddyn_correction = 0
        for (param_name, param), delta in zip(model.named_parameters(), delta_c):
            feddyn_correction += (alpha / 2) * torch.sum((param - global_state_dict[param_name]) * delta)

        # 总损失 = 分类损失 + FedDyn 校正项
        total_loss = loss + feddyn_correction

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * inputs.size(0)
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

# FedAvg 聚合函数 (包括FedDyn的全局更新)
def feddyn_aggregation(global_model, client_models, delta_c_global, client_delta_c):
    global_dict = global_model.state_dict()
    num_clients = len(client_models)

    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(num_clients)], 0).mean(0)
    
    # 更新全局控制变量
    for i in range(len(delta_c_global)):
        delta_c_global[i] += torch.stack([client_delta_c[j][i] for j in range(num_clients)], dim=0).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model, delta_c_global

# 初始化控制变量
def initialize_control_variates(model):
    control_variates = []
    for param in model.parameters():
        control_variates.append(torch.zeros_like(param))
    return control_variates

# 加载每个客户端的原始特征和标签文件
def load_client_features(client_idx, dataset_name, base_dir):
    features, labels = [], []
    for class_idx in range(10):
        original_features_path = os.path.join(base_dir, dataset_name, f'client_{client_idx}_class_{class_idx}/final_embeddings_filled.npy')
        original_labels_path = os.path.join(base_dir, dataset_name, f'client_{client_idx}_class_{class_idx}/labels_filled.npy')
        print(f"client_{client_idx}_class_{class_idx} 特征已经加载")
        
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
def load_test_features_labels(dataset_name, base_dir='./clip_test_features'):
    test_features_path = os.path.join(base_dir, dataset_name, f'{dataset_name}_test_features.npy')
    test_labels_path = os.path.join(base_dir, dataset_name, f'{dataset_name}_test_labels.npy')
    print(f"{dataset_name} 测试集特征已经加载")
    
    if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
        test_features = np.load(test_features_path)
        test_labels = np.load(test_labels_path)
        return torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)
    else:
        raise FileNotFoundError(f"{dataset_name} 的测试集特征或标签文件不存在")

# 联邦训练和评估函数 (FedDyn)
def federated_train_and_evaluate_feddyn(all_client_features, test_sets, communication_rounds, local_epochs, batch_size=16, learning_rate=0.001, alpha=0.5):
    global_model = MyNet(num_classes=10).to(device)
    client_models = [MyNet(num_classes=10).to(device) for _ in range(len(all_client_features))]

    criterion = nn.CrossEntropyLoss()

    delta_c_global = initialize_control_variates(global_model)
    client_delta_c = [initialize_control_variates(client_models[i]) for i in range(len(client_models))]

    report_path = './results/FedDyn_argumented_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    all_accuracies = {name: [] for name in list(test_sets.keys()) + ['average']}
    
    best_avg_accuracy = 0.0
    best_model_state = None

    with open(report_path, 'w') as f:
        for round in range(communication_rounds):
            for client_idx, client_data in enumerate(all_client_features):
                original_features, original_labels = client_data
                client_model = client_models[client_idx]
                optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)

                for epoch in range(local_epochs):
                    train_dataset = MyDataset(original_features, original_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    train_loss, train_accuracy = train_feddyn(client_model, global_model, client_delta_c[client_idx], train_dataloader, criterion, optimizer, device, alpha=alpha)

            global_model, delta_c_global = feddyn_aggregation(global_model, client_models, delta_c_global, client_delta_c)

            accuracies = []
            for dataset_name, (test_features, test_labels) in test_sets.items():
                test_dataset = MyDataset(test_features, test_labels)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                test_accuracy = test(global_model, test_dataloader, device)
                accuracies.append(test_accuracy)
                all_accuracies[dataset_name].append(test_accuracy)
            
            avg_accuracy = sum(accuracies) / len(accuracies)
            all_accuracies['average'].append(avg_accuracy)
            
            accuracy_output = (f"Round {round + 1}/{communication_rounds}, "
                               + ', '.join([f"{dataset_name.capitalize()} Accuracy: {test_accuracy:.4f}" for dataset_name, test_accuracy in zip(test_sets.keys(), accuracies)])
                               + f", Average Accuracy: {avg_accuracy:.4f}")
            
            print(accuracy_output)
            f.write(accuracy_output + "\n")
            
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_model_state = global_model.state_dict()

    return best_model_state, all_accuracies


# 客户端编号分配
client_range = {
    'mnist': [0],
    'usps': [1],
    'svhn': [2],
    'syn': [3]
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
    plt.savefig('./results/FedDyn_argumented__test_accuracy_plot.png')
    plt.show()

# 主函数
def main():
    base_dir = './argumented_clip_features'
    
    all_client_features = []
    datasets = ['mnist', 'usps', 'svhn', 'syn']
    for dataset_name in datasets:
        for client_id in client_range[dataset_name]:
            features, labels = load_client_features(client_id, dataset_name, base_dir)
            if features is not None and labels is not None:
                all_client_features.append((features, labels))

    test_sets = {}
    for dataset_name in datasets:
        test_features, test_labels = load_test_features_labels(dataset_name)
        test_sets[dataset_name] = (test_features, test_labels)
        
    os.makedirs('./model', exist_ok=True)

    communication_rounds = 50
    local_epochs = 10
    alpha = 0.5

    best_model_state, all_accuracies = federated_train_and_evaluate_feddyn(all_client_features, test_sets, communication_rounds, local_epochs, alpha=alpha)

    best_model_path = './model/FedDyn_argumented__best_model.pth'
    torch.save(best_model_state, best_model_path)

    plot_accuracies(all_accuracies, communication_rounds)

if __name__ == "__main__":
    main()
