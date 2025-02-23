import os
import numpy as np
import torch

def check_labels(client_idx, base_dir='./CIFAR-10/features'):
    for class_idx in range(10):
        original_labels_path = os.path.join(base_dir, 'initial', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels.npy')
        augmented_labels_path = os.path.join(base_dir, 'complete', f'alpha=0.1_class_{class_idx}_client_{client_idx}', 'labels_filled.npy')

        if os.path.exists(original_labels_path):
            original_labels = np.load(original_labels_path)
            print(f"Original labels for class {class_idx}: {np.unique(original_labels)}")
        else:
            print(f"Original labels file for class {class_idx}, client {client_idx} does not exist.")

        if os.path.exists(augmented_labels_path):
            augmented_labels = np.load(augmented_labels_path)
            print(f"Augmented labels for class {class_idx}: {np.unique(augmented_labels)}")
        else:
            print(f"Augmented labels file for class {class_idx}, client {client_idx} does not exist.")

if __name__ == "__main__":
    check_labels(client_idx=0)
