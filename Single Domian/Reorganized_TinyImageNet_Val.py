import os
import shutil
from collections import defaultdict

def load_class_map(datadir):
    wnid_map_file = os.path.join(datadir, 'wnids.txt')
    with open(wnid_map_file, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]

    class_map = {wnid: i for i, wnid in enumerate(wnids)}
    return class_map

def reorganize_val_images(datadir):
    annotations_file = os.path.join(datadir, 'val', 'val_annotations.txt')
    val_images_dir = os.path.join(datadir, 'val', 'images')
    new_val_dir = os.path.join(datadir, 'new_val')

    # 创建new_val目录
    os.makedirs(new_val_dir, exist_ok=True)

    # 加载类映射
    class_map = load_class_map(datadir)

    # 创建每个类的文件夹
    for wnid in class_map.keys():
        class_dir = os.path.join(new_val_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)

    # 读取annotations文件并移动图片
    with open(annotations_file, 'r') as f:
        annotations = f.readlines()

    for line in annotations:
        parts = line.strip().split('\t')
        image_name = parts[0]
        label = parts[1]
        src_path = os.path.join(val_images_dir, image_name)
        dst_path = os.path.join(new_val_dir, label, image_name)
        shutil.move(src_path, dst_path)

    print(f"Reorganized validation images saved to {new_val_dir}")

def main():
    datadir = './data/tiny-imagenet-200'
    reorganize_val_images(datadir)

if __name__ == "__main__":
    main()
