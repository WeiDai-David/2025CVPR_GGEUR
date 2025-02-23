import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from PIL import Image
import warnings
import open_clip
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
    
    
# 计算并保存类别权重范数图像和TXT文件
def calculate_and_save_weight_norms(model, round_num, alpha, client_num=None, save_global=False, feature_type="original"):
    layer_weights = model.fc3.weight.data.cpu().numpy()  # 获取fc3层的权重
    weight_norms = np.linalg.norm(layer_weights, axis=1)
    sorted_indices = np.argsort(-weight_norms)
    sorted_weight_norms = weight_norms[sorted_indices]

    plt.figure()
    plt.plot(sorted_weight_norms)
    plt.xlabel('Classes (sorted by weight norm)')
    plt.ylabel('Weight Norm')

    if save_global:
        plt.title(f'Global Model Round {round_num}')
        save_dir = f'./CIFAR-10/clip_fedavg_alpha_{alpha}/{feature_type}/global'
    else:
        plt.title(f'Client {client_num} Round {round_num}')
        save_dir = f'./CIFAR-10/clip_fedavg_alpha_{alpha}/{feature_type}/client{client_num}'
    
    os.makedirs(save_dir, exist_ok=True)
    
    if save_global:
        plt.savefig(os.path.join(save_dir, f'global_round_{round_num}.png'))
        txt_path = os.path.join(save_dir, f'global_round_{round_num}_norms.txt')
    else:
        plt.savefig(os.path.join(save_dir, f'client{client_num}_round_{round_num}.png'))
        txt_path = os.path.join(save_dir, f'client{client_num}_round_{round_num}_norms.txt')

    plt.close()

    with open(txt_path, 'w') as f:
        for idx, norm in zip(sorted_indices, sorted_weight_norms):
            f.write(f'Class {idx}: Norm {norm}\n')

    return max(weight_norms) / min(weight_norms)


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

# 解析 CIFAR-10 数据集的批次文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载客户端特征文件
def load_client_features(client_idx, base_dir, alpha):
    features_dict = {'original': {}, 'augmented': {}}
    for class_idx in range(10):
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
alpha = 0.1  # 默认alpha值
communication_rounds = 30  # 默认通信轮数
local_epochs = 10  # 默认每轮通信的本地epoch数
client_count = 10  
base_dir = './CIFAR-10/features'

# 加载所有客户端的原始和补全特征文件
all_client_features = []
for client_idx in range(client_count):
    client_features_dict = load_client_features(client_idx, base_dir, alpha)
    original_features, original_labels = merge_features(client_features_dict['original'])
    augmented_features, augmented_labels = merge_features(client_features_dict['augmented'])
    all_client_features.append((original_features, original_labels, augmented_features, augmented_labels))

# 打印第一个客户端的特征和标签的大小和内容示例
print("第一个客户端原始特征大小:", all_client_features[0][0].shape)
print("第一个客户端原始标签大小:", all_client_features[0][1].shape)
print("第一个客户端补全特征大小:", all_client_features[0][2].shape)
print("第一个客户端补全标签大小:", all_client_features[0][3].shape)

# 加载预训练的 CLIP 模型
backbone = 'ViT-B-32'
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'
clip_model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
clip_model.eval().to(device)

# 加载 CIFAR-10 测试数据并提取特征
test_features_dir = f'./CIFAR-10/features/alpha={alpha}_test'
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
    test_data_batch_file = './data/cifar-10-batches-py/test_batch'
    test_data_batch = unpickle(test_data_batch_file)
    test_data = test_data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
    test_labels = np.array(test_data_batch[b'labels'])

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
    return accuracy

# 联邦学习过程中的聚合函数
def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# 训练和评估函数
def federated_train_and_evaluate(all_client_features, test_features, test_labels, communication_rounds, local_epochs, alpha, batch_size=64, learning_rate=0.001):
    
    feature_type = "original"
    
    global_model = MyNet(num_classes=10).to(device)
    client_models = [MyNet(num_classes=10).to(device) for _ in range(client_count)]

    criterion = nn.CrossEntropyLoss()
    test_accuracies = []
    client_norm_ratios = [[] for _ in range(client_count)]
    global_norm_ratios = []

    report_path = f'./CIFAR-10/clip_fedavg_alpha_{alpha}/report.txt'

    with open(report_path, 'w') as f:
        for round in range(communication_rounds):
            for client_idx, client_data in enumerate(all_client_features):
                original_features, original_labels, augmented_features, augmented_labels = client_data
                client_model = client_models[client_idx]
                optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)
                
                best_local_model = None
                best_local_accuracy = 0

                for epoch in range(local_epochs):
                    train_dataset = MyDataset(original_features, original_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    train_loss, train_accuracy = train(client_model, train_dataloader, criterion, optimizer, device)

                    valid_dataset = MyDataset(original_features, original_labels)
                    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                    valid_accuracy, _, _ = test(client_model, valid_dataloader, device)
                    if valid_accuracy > best_local_accuracy:
                        best_local_accuracy = valid_accuracy
                        best_local_model = client_model.state_dict()

                client_models[client_idx].load_state_dict(best_local_model)

                # 计算并保存客户端的fc3层权重范数
                client_norm_ratio = calculate_and_save_weight_norms(client_model, round, alpha, client_num=client_idx, feature_type=feature_type)
                client_norm_ratios[client_idx].append(client_norm_ratio)

            # FedAvg聚合
            global_model = federated_averaging(global_model, client_models)

            # 计算并保存全局模型的fc3层权重范数
            global_norm_ratio = calculate_and_save_weight_norms(global_model, round, alpha, save_global=True, feature_type=feature_type)
            global_norm_ratios.append(global_norm_ratio)

            test_dataset = MyDataset(test_features.numpy(), test_labels.numpy())
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_accuracy, _, _ = test(global_model, test_dataloader, device)
            test_accuracies.append(test_accuracy * 100)
            print(f"Round {round + 1}/{communication_rounds}, Test Accuracy: {test_accuracy:.4f}")
            f.write(f"Round {round + 1}/{communication_rounds}, Test Accuracy: {test_accuracy:.4f}\n")

    return global_model, test_accuracies, client_norm_ratios, global_norm_ratios


# 设置训练参数
batch_size = 64
learning_rate = 0.001

# 训练和评估原始特征模型
original_model, original_test_accuracies, client_norm_ratios, global_norm_ratios = federated_train_and_evaluate(all_client_features, test_features, test_labels, communication_rounds, local_epochs, alpha)

# 绘制客户端和全局模型的最大/最小权重范数比值变化图像
for idx in range(client_count):
    plt.figure()
    plt.plot(range(communication_rounds), client_norm_ratios[idx], label=f'Client {idx}')
    plt.plot(range(communication_rounds), global_norm_ratios, label='Global')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Max/Min Weight Norm Ratio')
    plt.legend()
    plt.title(f'Client {idx} and Global Model Norm Ratios')
    plt.savefig(f'./CIFAR-10/clip_fedavg_alpha_{alpha}/original/client{idx}_global_norm_ratios.png')
    plt.close()

# 保存原始特征模型
original_model_path = f'./CIFAR-10/clip_fedavg_alpha_{alpha}/model/global_original_model.pth'
torch.save(original_model.state_dict(), original_model_path)

# 绘制原始特征模型精度随通信轮数变化图
plt.figure()
plt.plot(range(1, communication_rounds + 1), original_test_accuracies, label='Original Features')
plt.xlabel('Communication Rounds')
plt.ylabel('Top-1 Accuracy (%)')
plt.ylim(0, 100)
plt.title('Top-1 Accuracy vs Communication Rounds')
plt.legend()
plt.savefig(f'./CIFAR-10//clip_fedavg_alpha_{alpha}/model/original_test_accuracies_0_100.png')

plt.figure()
plt.plot(range(1, communication_rounds + 1), original_test_accuracies, label='Original Features')
plt.xlabel('Communication Rounds')
plt.ylabel('Top-1 Accuracy (%)')
plt.title('Top-1 Accuracy vs Communication Rounds')
plt.legend()
plt.savefig(f'./CIFAR-10/clip_fedavg_alpha_{alpha}/original_test_accuracies.png')

plt.show()

# 保存测试精度最高的原始特征模型
best_original_accuracy = max(original_test_accuracies)
best_original_round = original_test_accuracies.index(best_original_accuracy) + 1
print(f"Best Original Features Model Accuracy: {best_original_accuracy:.2f}% at Communication Round {best_original_round}")

# 训练和评估补全特征模型
def federated_train_and_evaluate_augmented(all_client_features, test_features, test_labels, communication_rounds, local_epochs, alpha, batch_size=64, learning_rate=0.001):
   
    feature_type = "augmented"  # 设置为补全特征
    client_norm_ratios = [[] for _ in range(client_count)]
    global_norm_ratios = []
    
    global_model = MyNet(num_classes=10).to(device)
    client_models = [MyNet(num_classes=10).to(device) for _ in range(client_count)]

    criterion = nn.CrossEntropyLoss()

    test_accuracies = []
    report_path = f'./CIFAR-10/clip_fedavg_alpha_{alpha}/model/report_augmented.txt'

    with open(report_path, 'w') as f:
        for round in range(communication_rounds):
            for client_idx, client_data in enumerate(all_client_features):
                original_features, original_labels, augmented_features, augmented_labels = client_data
                client_model = client_models[client_idx]
                optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)
                
                best_local_model = None
                best_local_accuracy = 0

                for epoch in range(local_epochs):
                    train_dataset = MyDataset(augmented_features, augmented_labels)  # 使用补全特征训练
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    train_loss, train_accuracy = train(client_model, train_dataloader, criterion, optimizer, device)

                    # 在本地验证集上评估，以选择最优模型
                    valid_dataset = MyDataset(augmented_features, augmented_labels)
                    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                    valid_accuracy, _, _ = test(client_model, valid_dataloader, device)
                    if valid_accuracy > best_local_accuracy:
                        best_local_accuracy = valid_accuracy
                        best_local_model = client_model.state_dict()

                # 使用最优模型参数更新全局模型
                client_models[client_idx].load_state_dict(best_local_model)
                
                client_norm_ratio = calculate_and_save_weight_norms(client_model, round, alpha, client_num=client_idx, feature_type=feature_type)
                client_norm_ratios[client_idx].append(client_norm_ratio)
            # FedAvg聚合
            global_model = federated_averaging(global_model, client_models)
            
            # 计算并保存全局模型的fc3层权重范数
            global_norm_ratio = calculate_and_save_weight_norms(global_model, round, alpha, save_global=True, feature_type=feature_type)
            global_norm_ratios.append(global_norm_ratio)

            # 在测试集上评估聚合后的全局模型
            test_dataset = MyDataset(test_features.numpy(), test_labels.numpy())
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_accuracy, _, _ = test(global_model, test_dataloader, device)
            test_accuracies.append(test_accuracy * 100)  # 转换为百分比形式
            print(f"Communication Round {round + 1}/{communication_rounds}, Test Accuracy: {test_accuracy:.4f}")
            f.write(f"Communication Round {round + 1}/{communication_rounds}, Test Accuracy: {test_accuracy:.4f}\n")

    return global_model, test_accuracies, client_norm_ratios, global_norm_ratios

# 训练和评估补全特征模型
augmented_model, augmented_test_accuracies, client_norm_ratios, global_norm_ratios = federated_train_and_evaluate_augmented(all_client_features, test_features, test_labels, communication_rounds, local_epochs, alpha)

# 绘制客户端和全局模型的最大/最小权重范数比值变化图像
for idx in range(client_count):
    plt.figure()
    plt.plot(range(communication_rounds), client_norm_ratios[idx], label=f'Client {idx}')
    plt.plot(range(communication_rounds), global_norm_ratios, label='Global')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Max/Min Weight Norm Ratio')
    plt.legend()
    plt.title(f'Client {idx} and Global Model Norm Ratios')
    plt.savefig(f'./CIFAR-10/clip_fedavg_alpha_{alpha}/augmented/client{idx}_global_norm_ratios.png')
    plt.close()
    
# 保存补全特征模型
augmented_model_path = f'./CIFAR-10/clip_fedavg_alpha_{alpha}/model/global_augmented_model.pth'
torch.save(augmented_model.state_dict(), augmented_model_path)

# 绘制补全特征模型精度随通信轮数变化图
plt.figure()
plt.plot(range(1, communication_rounds + 1), augmented_test_accuracies, label='Augmented Features')
plt.xlabel('Communication Rounds')
plt.ylabel('Top-1 Accuracy (%)')
plt.ylim(0, 100)
plt.title('Top-1 Accuracy vs Communication Rounds')
plt.legend()
plt.savefig(f'./CIFAR-10/clip_fedavg_alpha_{alpha}/model/augmented_test_accuracies_0_100.png')

plt.figure()
plt.plot(range(1, communication_rounds + 1), augmented_test_accuracies, label='Augmented Features')
plt.xlabel('Communication Rounds')
plt.ylabel('Top-1 Accuracy (%)')
plt.title('Top-1 Accuracy vs Communication Rounds')
plt.legend()
plt.savefig(f'./CIFAR-10/clip_fedavg_alpha_{alpha}/model/augmented_test_accuracies.png')

plt.show()


# 保存测试精度最高的补全特征模型
best_augmented_accuracy = max(augmented_test_accuracies)
best_augmented_round = augmented_test_accuracies.index(best_augmented_accuracy) + 1
print(f"Best Augmented Features Model Accuracy: {best_augmented_accuracy:.2f}% at Communication Round {best_augmented_round}")


