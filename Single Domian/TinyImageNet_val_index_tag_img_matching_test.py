import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# 定义数据集类
class TinyImageNet_truncated:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = ImageFolder(root=self.root, transform=self.transform)
        self.data, self.targets = zip(*[(data, target) for data, target in self.dataset])
        self.data = list(self.data)
        self.targets = list(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

# 加载验证集数据
def load_tinyimagenet_val_data(datadir):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    val_dataset = TinyImageNet_truncated(root=f'{datadir}/new_val', transform=transform)
    X_val, y_val = val_dataset.data, val_dataset.targets
    return (X_val, y_val)

# 显示图像和对应标签，并保存图像
def display_and_save_image(index, X_val, y_val, class_map, save_path):
    img = X_val[index]
    label = y_val[index]

    # 获取类别名
    class_name = class_map[label]

    # 将图像从 Tensor 转换回 PIL Image
    img = transforms.ToPILImage()(img)

    # 显示图像和标签
    plt.imshow(img)
    plt.title(f'Index: {index}, Label: {label}, Class: {class_name}')
    plt.savefig(save_path)
    plt.close()

def load_class_mapping(output_dir):
    class_map_file = os.path.join(output_dir, 'class_map.txt')
    class_map = {}
    with open(class_map_file, 'r') as f:
        for line in f:
            idx, wnid = line.strip().split('\t')
            class_map[int(idx)] = wnid
    return class_map

def main():
    parser = argparse.ArgumentParser(description='Display and save image from TinyImageNet validation set.')
    parser.add_argument('--index', type=int, default= 0, help='Index of the image to display and save.')
    parser.add_argument('--save_path', type=str, default='./TinyImageNet/images/Index-tag-image-matching_test.jpg', help='Path to save the displayed image.')
    args = parser.parse_args()

    datadir = './data/tiny-imagenet-200'
    output_dir = "./TinyImageNet/val_context"
    X_val, y_val = load_tinyimagenet_val_data(datadir)
    class_map = load_class_mapping(output_dir)

    if args.index < 0 or args.index >= len(X_val):
        print(f"Index out of range. Please enter an index between 0 and {len(X_val) - 1}.")
        return

    display_and_save_image(args.index, X_val, y_val, class_map, args.save_path)
    print(f"Image at index {args.index} saved to {args.save_path}")

if __name__ == "__main__":
    main()
