import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
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
    def __init__(self, num_classes=100):
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

def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        print(f"Evaluating model: features.shape={features.shape}, labels.shape={labels.shape}")
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        print(f"Evaluation: predicted.shape={predicted.shape}, labels.shape={labels.shape}")
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        report = classification_report(labels.cpu().numpy(), predicted.cpu().numpy(), digits=4)
    return accuracy, report

def load_features(feature_dir, alpha, num_classes, is_filled=False):
    features = []
    labels = []
    for class_idx in range(num_classes):
        for client_idx in range(10):
            feature_file = 'final_embeddings_filled.npy' if is_filled else 'final_embeddings.npy'
            feature_path = os.path.join(feature_dir, f'alpha={alpha}_class_{class_idx}_client_{client_idx}', feature_file)
            label_path = os.path.join(feature_dir, f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'labels.npy') if not is_filled else os.path.join(feature_dir, f'alpha={alpha}_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')
            if os.path.exists(feature_path) and os.path.exists(label_path):
                features.append(np.load(feature_path))
                labels.append(np.load(label_path))
    features = np.vstack(features)
    labels = np.concatenate(labels)
    print(f"Loaded features: {features.shape}, labels: {labels.shape}")
    return features, labels

def initialize_global_model(num_classes):
    model = MyNet(num_classes=num_classes).to(device)
    return model

def train_on_client(features, labels, global_model_path, num_classes, num_epochs=10):
    model = MyNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(global_model_path))

    train_dataset = MyDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_accuracy = 0.0
    best_model_path = None

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 每个 epoch 后进行测试
        accuracy, _ = evaluate_model(model, torch.tensor(features).to(device), torch.tensor(labels).to(device))
        print(f"Client Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = f'client_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), best_model_path)

    return best_model_path

def aggregate_client_models(client_model_paths, global_model):
    global_state_dict = global_model.state_dict()

    avg_state_dict = {key: torch.zeros_like(value) for key, value in global_state_dict.items()}

    for client_model_path in client_model_paths:
        client_state_dict = torch.load(client_model_path)
        for key in avg_state_dict:
            avg_state_dict[key] += client_state_dict[key]

    for key in avg_state_dict:
        avg_state_dict[key] /= len(client_model_paths)

    global_model.load_state_dict(avg_state_dict)
    return global_model

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

def save_features_and_labels(features, labels, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    np.save(os.path.join(base_dir, 'final_embeddings.npy'), features)
    np.save(os.path.join(base_dir, 'labels.npy'), labels)

def load_saved_test_features_and_labels(base_dir):
    features_path = os.path.join(base_dir, 'final_embeddings.npy')
    labels_path = os.path.join(base_dir, 'labels.npy')
    if os.path.exists(features_path) and os.path.exists(labels_path):
        features = np.load(features_path)
        labels = np.load(labels_path)
        print(f"Loaded test features: {features.shape}, labels: {labels.shape}")
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    return None, None

def main():
    num_rounds = 50
    num_epochs = 10
    alpha = 0.5
    num_classes = 100
    client_ids = list(range(10))
    test_features_dir = f'./CIFAR-100/features/alpha={alpha}_test'
    output_dir = f'./CIFAR-100/features/alpha={alpha}_model/Fedavg_num_rounds={num_rounds}_epoch={num_epochs}'
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否存在已保存的测试特征和标签
    test_features, test_labels = load_saved_test_features_and_labels(test_features_dir)

    if test_features is None or test_labels is None:
        # 加载测试数据
        print("Creating test features... Creating test labels...")
        test_data_batch_file = './data/cifar-100-python/test'
        test_data_batch = unpickle(test_data_batch_file)
        test_data = test_data_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为图像格式
        test_labels = np.array(test_data_batch[b'fine_labels'])

        backbone = 'ViT-B-32'
        pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'
        clip_model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
        clip_model.eval().to(device)

        test_indices = np.arange(len(test_data))
        test_features = extract_image_features(clip_model, test_data, test_indices, preprocess)
        test_labels = torch.tensor(test_labels)

        # 保存测试特征和标签
        save_features_and_labels(test_features.numpy(), test_labels.numpy(), test_features_dir)

    print(f"测试集图像数量: {test_features.shape[0]}")
    print(f"测试集标签数量: {test_labels.shape[0]}")
    print(f"提取的测试集特征数量: {test_features.shape[0]}")
    print(f"测试集标签数量: {test_labels.shape[0]}")

    def run_federated_learning(global_model, features_dir, is_filled, model_label):
        global_model = initialize_global_model(num_classes)
        torch.save(global_model.state_dict(), f'{model_label}_global_initial_model.pth')

        accuracies = []
        best_global_accuracy = 0.0
        best_global_model_path = None

        for round in range(num_rounds):
            print(f"Round {round + 1} for {model_label} features")

            client_model_paths = []
            for client_id in client_ids:
                features, labels = load_features(features_dir, alpha, num_classes, is_filled)
                best_client_model_path = train_on_client(features, labels, f'{model_label}_global_initial_model.pth', num_classes, num_epochs)
                client_model_paths.append(best_client_model_path)

            global_model = aggregate_client_models(client_model_paths, global_model)
            round_dir = f'{output_dir}/recent_round_{round + 1}'
            os.makedirs(round_dir, exist_ok=True)
            torch.save(global_model.state_dict(), os.path.join(round_dir, f'{model_label}_global_model.pth'))

            accuracy, report = evaluate_model(global_model, test_features, test_labels)
            print(f"Round {round + 1} - {model_label} feature model's CIFAR-100 test accuracy: {accuracy * 100:.2f}%")
            print(f"Round {round + 1} - {model_label} feature model's CIFAR-100 classification report:\n{report}")
            accuracies.append(accuracy * 100)
            with open(os.path.join(round_dir, f'{model_label}_report.txt'), 'w') as f:
                f.write(f"{model_label} feature model's CIFAR-100 test accuracy: {accuracy * 100:.2f}%\n")
                f.write(f"{model_label} feature model's CIFAR-100 classification report:\n{report}\n")

            # 保存最好的聚合模型
            if accuracy > best_global_accuracy:
                best_global_accuracy = accuracy
                best_global_model_path = os.path.join(output_dir, f'{model_label}_best_global_model.pth')
                torch.save(global_model.state_dict(), best_global_model_path)

        print(f"Best {model_label} Global Model Accuracy: {best_global_accuracy * 100:.2f}%")
        torch.save(global_model.state_dict(), f'{model_label}_global_model.pth')

        return accuracies, best_global_model_path

    # 联邦学习原始特征
    original_accuracies, best_original_global_model_path = run_federated_learning(initialize_global_model(num_classes), './CIFAR-100/features/initial', False, 'original')

    # 联邦学习补全特征
    augmented_accuracies, best_augmented_global_model_path = run_federated_learning(initialize_global_model(num_classes), f'./CIFAR-100/features/alpha={alpha}_complete', True, 'augmented')

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), original_accuracies, label='Original Features')
    plt.plot(range(1, num_rounds + 1), augmented_accuracies, label='Augmented Features')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title(f'FedAvg on CIFAR-100 {alpha}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'fedavg_accuracy_plot_percentage.png'))
    plt.show()

    # 绘制0-100区间的图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), original_accuracies, label='Original Features')
    plt.plot(range(1, num_rounds + 1), augmented_accuracies, label='Augmented Features')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.ylim(0, 100)
    plt.title(f'FedAvg on CIFAR-100 {alpha} (0-100 Range)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fedavg_accuracy_plot_0_100.png'))
    plt.show()

if __name__ == "__main__":
    main()
