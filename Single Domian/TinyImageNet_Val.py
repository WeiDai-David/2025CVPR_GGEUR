import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 定义数据集类
class TinyImageNet_truncated:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = ImageFolder(root=self.root, transform=self.transform)
        self.data, self.targets = zip(*[(data, target) for data, target in self.dataset])
        self.data = list(self.data)
        self.targets = list(self.targets)
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, target) in enumerate(self.dataset):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)
        return class_indices

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

# 加载 TinyImageNet 验证集数据
def load_tinyimagenet_val_data(datadir):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    val_dataset = TinyImageNet_truncated(root=f'{datadir}/new_val', transform=transform)
    X_val, y_val, class_indices = val_dataset.data, val_dataset.targets, val_dataset.class_indices
    return (X_val, y_val, class_indices)

# 生成类映射
def generate_class_mapping(new_val_dir):
    class_dirs = [d for d in os.listdir(new_val_dir) if os.path.isdir(os.path.join(new_val_dir, d))]
    class_dirs_sorted = sorted(class_dirs, key=lambda x: int(x[1:]))  # 去除 'n' 后按数值排序
    class_map = {i: class_dirs_sorted[i] for i in range(len(class_dirs_sorted))}
    return class_map

# 保存类映射到文件
def save_class_mapping(class_map, output_dir):
    class_map_file = os.path.join(output_dir, 'class_map.txt')
    with open(class_map_file, 'w') as f:
        for idx, wnid in class_map.items():
            f.write(f"{idx}\t{wnid}\n")
    print(f"Class mapping saved to {class_map_file}")

def main():
    datadir = './data/tiny-imagenet-200'
    new_val_dir = os.path.join(datadir, 'new_val')
    X_val, y_val, val_class_indices = load_tinyimagenet_val_data(datadir)

    # 创建输出目录
    output_dir = "./TinyImageNet/val_context"
    os.makedirs(output_dir, exist_ok=True)

    # 生成类映射
    class_map = generate_class_mapping(new_val_dir)

    # 保存验证集标签
    np.save(os.path.join(output_dir, "val_labels.npy"), y_val)
    
    with open(os.path.join(output_dir, "val_labels.txt"), "w") as f:
        for label in y_val:
            f.write(f"{label}\n")

    # 保存验证集索引
    indices = list(range(len(X_val)))
    np.save(os.path.join(output_dir, "val_indices.npy"), indices)
    
    with open(os.path.join(output_dir, "val_indices.txt"), "w") as f:
        for idx in range(len(X_val)):
            f.write(f"{idx}\n")

    # 保存每个类的索引
    np.set_printoptions(threshold=np.inf)  # 设置打印选项为显示所有值
    for class_label, indices in val_class_indices.items():
        np.save(os.path.join(f'./TinyImageNet/class_{class_label}_val_indices.npy'), indices)  # 将索引保存到文件
        # print(f'Class {class_label} indices: {indices}')
    np.set_printoptions(threshold=1000)  # 恢复默认打印选项

    # 保存类编号到TinyImageNet编号的映射
    save_class_mapping(class_map, output_dir)

if __name__ == "__main__":
    main()
