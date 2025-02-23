# 根据100个原始特征文件所在文件夹，生成100个原始特征文件对应的标签文件,具体的原始特征以该类下该客户端样本数*512的矩阵形式，而矩阵的一行代表一个样本，对应一个标签
import numpy as np
import os

def generate_labels():
    base_dir = './CIFAR-10/features/'
    for class_idx in range(10):
        for client_idx in range(10):
            features_path = os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'final_embeddings.npy')
            if os.path.exists(features_path):
                class_features = np.load(features_path)
                labels = np.full(class_features.shape[0], class_idx)
                labels_dir = os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels.npy')
                os.makedirs(os.path.dirname(labels_dir), exist_ok=True)
                np.save(labels_dir, labels)
                print(f"Saved labels for class {class_idx}, client {client_idx}")

if __name__ == "__main__":
    generate_labels()



