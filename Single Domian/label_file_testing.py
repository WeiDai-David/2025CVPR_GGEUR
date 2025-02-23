# label_file_testing.py 该脚本可以测算标签文件的标签数量,以及包含的索引有哪些(正常情况下一个class只能有一个)
# 当前选择对客户端0的initial和complete的标签文件进行操作
import os
import numpy as np
import torch

def check_labels(client_idx, base_dir='./CIFAR-10/features'):
    original_labels_list = []
    augmented_labels_list = []

    for class_idx in range(10):
        original_labels_path = os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels.npy')
        augmented_labels_path = os.path.join(base_dir, 'complete', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')

        if os.path.exists(original_labels_path):
            original_labels = np.load(original_labels_path)
            if original_labels.size > 0:
                original_labels_list.append(original_labels)
            print(f"Original labels for class {class_idx}: {np.unique(original_labels)}, Sample size: {original_labels.size}")
        else:
            print(f"Original labels file for class {class_idx}, client {client_idx} does not exist.")

        if os.path.exists(augmented_labels_path):
            augmented_labels = np.load(augmented_labels_path)
            if augmented_labels.size > 0:
                augmented_labels_list.append(augmented_labels)
            print(f"Augmented labels for class {class_idx}: {np.unique(augmented_labels)}, Sample size: {augmented_labels.size}")
        else:
            print(f"Augmented labels file for class {class_idx}, client {client_idx} does not exist.")

    # 检查哪些类有样本，哪些类没有样本
    classes_with_samples = [class_idx for class_idx in range(10) if np.load(os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels.npy')).size > 0]
    classes_without_samples = [class_idx for class_idx in range(10) if np.load(os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels.npy')).size == 0]

    print(f"Classes with samples: {classes_with_samples}")
    print(f"Classes without samples: {classes_without_samples}")

if __name__ == "__main__":
    check_labels(client_idx=0)
