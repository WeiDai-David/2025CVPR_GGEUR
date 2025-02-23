import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import copy
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
    def __init__(self, input_dim=512, num_classes=7):
        super(MyNet, self).__init__()
        self.fc3 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return F.softmax(self.fc3(x), dim=1)

# 训练函数，包含FedProto的Proximal weight λ项
def train_fedproto(model, dataloader, criterion, optimizer, device, global_protos, local_protos, lam=2):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    agg_protos_label = {}
    protos_label_num = {}
    mse_loss = nn.MSELoss()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 获取模型输出
        outputs = model(inputs)
        loss_ce = criterion(outputs, labels)

        # 原型损失
        if global_protos:
            new_features = inputs.clone().detach()  # 直接使用 CLIP 提取的特征
            for i, label in enumerate(labels):
                if label.item() in global_protos:
                    new_features[i] = global_protos[label.item()][0]
            loss_proto = mse_loss(new_features, inputs) * lam
        else:
            loss_proto = torch.tensor(0.0, device=device)

        # 总损失
        loss = loss_ce + loss_proto
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新本地原型
        with torch.no_grad():
            for i in range(len(labels)):
                label = labels[i].item()
                if label in agg_protos_label:
                    agg_protos_label[label] += inputs[i].detach()  # 直接使用 CLIP 特征
                    protos_label_num[label] += 1
                else:
                    agg_protos_label[label] = inputs[i].detach().clone()
                    protos_label_num[label] = 1

    # 计算每个类的本地原型
    for label in agg_protos_label:
        agg_protos_label[label] = agg_protos_label[label] / protos_label_num[label]
    local_protos.update(agg_protos_label)

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

# FedProto 聚合函数
def fedproto_aggregation(global_model, client_models, global_protos, local_protos_list):
    # 平均聚合模型参数
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

    # 聚合客户端的类原型
    new_global_protos = {}
    for local_protos in local_protos_list:
        for label, proto in local_protos.items():
            if label not in new_global_protos:
                new_global_protos[label] = [proto]
            else:
                new_global_protos[label].append(proto)

    # 计算全局原型的平均
    for label, proto_list in new_global_protos.items():
        new_global_protos[label] = torch.stack(proto_list, 0).mean(0)

    global_protos.update(new_global_protos)
    return global_model, global_protos


# 加载训练集特征
def load_training_data(dataset_name, client_id, base_dir):
    all_features = []
    all_labels = []

    for class_idx in range(7):
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

# 联邦训练和评估函数 (FedProto)
def federated_train_and_evaluate_fedproto(all_client_features, test_loaders, communication_rounds, local_epochs, batch_size=16, learning_rate=0.01, lam=2):
    global_model = MyNet(num_classes=7).to(device)
    client_models = [MyNet(num_classes=7).to(device) for _ in range(len(all_client_features))]
    
    criterion = nn.CrossEntropyLoss()
    global_protos = {}
    local_protos_list = [{} for _ in range(len(all_client_features))]

    test_accuracies_photo = []
    test_accuracies_art_painting = []
    test_accuracies_cartoon = []
    test_accuracies_sketch = []
    avg_accuracies = []

    best_avg_accuracy = 0.0
    best_model_state = None

    report_path = './results/FedProto_original_report.txt'
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
                    train_loss, train_accuracy = train_fedproto(client_model, train_dataloader, criterion, optimizer, device, global_protos, local_protos_list[client_idx], lam=lam)

            # 聚合
            global_model, global_protos = fedproto_aggregation(global_model, client_models, global_protos, local_protos_list)
            
            # 在四个测试集上分别评估全局模型
            photo_features, photo_labels = test_loaders['photo']
            art_painting_features, art_painting_labels = test_loaders['art_painting']
            cartoon_features, cartoon_labels = test_loaders['cartoon']
            sketch_features, sketch_labels = test_loaders['sketch']

            test_accuracy_photo = test(global_model, DataLoader(MyDataset(photo_features, photo_labels), batch_size=batch_size), device)
            test_accuracy_art_painting = test(global_model, DataLoader(MyDataset(art_painting_features, art_painting_labels), batch_size=batch_size), device)
            test_accuracy_cartoon = test(global_model, DataLoader(MyDataset(cartoon_features, cartoon_labels), batch_size=batch_size), device)
            test_accuracy_sketch = test(global_model, DataLoader(MyDataset(sketch_features, sketch_labels), batch_size=batch_size), device)

            avg_accuracy = (test_accuracy_photo + test_accuracy_art_painting + test_accuracy_cartoon + test_accuracy_sketch) / 4

            test_accuracies_photo.append(test_accuracy_photo * 100)
            test_accuracies_art_painting.append(test_accuracy_art_painting * 100)
            test_accuracies_cartoon.append(test_accuracy_cartoon * 100)
            test_accuracies_sketch.append(test_accuracy_sketch * 100)
            avg_accuracies.append(avg_accuracy * 100)

            print(f"Communication Round {round + 1}/{communication_rounds}, Photo Accuracy: {test_accuracy_photo:.4f}, Art Painting Accuracy: {test_accuracy_art_painting:.4f}, Cartoon Accuracy: {test_accuracy_cartoon:.4f}, Sketch Accuracy: {test_accuracy_sketch:.4f}, Average Accuracy: {avg_accuracy:.4f}")
            f.write(f"Communication Round {round + 1}/{communication_rounds}, Photo Accuracy: {test_accuracy_photo:.4f}, Art Painting Accuracy: {test_accuracy_art_painting:.4f}, Cartoon Accuracy: {test_accuracy_cartoon:.4f}, Sketch Accuracy: {test_accuracy_sketch:.4f}, Average Accuracy: {avg_accuracy:.4f}\n")

            # 如果当前的平均精度超过之前的最佳精度，则保存当前模型
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_model_state = global_model.state_dict()  # 保存最佳模型的状态

    # 返回最佳模型状态
    return best_model_state, test_accuracies_photo, test_accuracies_art_painting, test_accuracies_cartoon, test_accuracies_sketch, avg_accuracies

# 主函数
def main():
    
    # 客户端编号分配
    client_range = {
        'photo': [0],
        'art_painting': [1],
        'cartoon': [2],
        'sketch': [3]
    }
    # 加载客户端-数据集映射
    dataset_clients = client_range

    # 加载训练集客户端的特征和标签
    train_base_dir = './clip_pacs_train_features'
    all_client_features = []
    for dataset_name, clients in dataset_clients.items():
        for client_id in clients:
            features, labels = load_training_data(dataset_name, client_id, train_base_dir)
            all_client_features.append((features, labels))

    # 加载测试集特征和标签
    test_base_dir = './clip_pacs_test_features'
    test_loaders = {
        'photo': load_test_data('photo', test_base_dir),
        'art_painting': load_test_data('art_painting', test_base_dir),
        'cartoon': load_test_data('cartoon', test_base_dir),
        'sketch': load_test_data('sketch', test_base_dir)
    }

    # 联邦训练和评估
    communication_rounds = 50
    local_epochs = 10
    best_model_state, test_accuracies_photo, test_accuracies_art_painting, test_accuracies_cartoon, test_accuracies_sketch, avg_accuracies = federated_train_and_evaluate_fedproto(
        all_client_features, test_loaders, communication_rounds, local_epochs, lam=2)

    # 保存精度最高的模型
    best_model_path = './model/FedProto_original_best_model.pth'
    torch.save(best_model_state, best_model_path)

    # 绘制测试集精度随通信轮次变化的折线图
    plt.figure()
    plt.plot(range(1, communication_rounds + 1), test_accuracies_photo, label='Photo Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_art_painting, label='Art Painting Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_cartoon, label='Cartoon Accuracy')
    plt.plot(range(1, communication_rounds + 1), test_accuracies_sketch, label='Sketch Accuracy')
    plt.plot(range(1, communication_rounds + 1), avg_accuracies, label='Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(0, 100)
    plt.title('Test Accuracy vs Communication Rounds')
    plt.legend()
    plt.savefig('./results/FedProto_original_test_accuracy_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
