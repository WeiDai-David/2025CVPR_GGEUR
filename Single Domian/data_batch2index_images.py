# 这个脚本根据划分后保存的索引，用来检验是否划分正确
import pickle
import numpy as np
from PIL import Image
import os

# 加载 CIFAR 数据集
def load_cifar_batch(file, dataset='cifar10'):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels'] if dataset == 'cifar10' else datadict[b'fine_labels']
        if dataset == 'cifar10':
            X = X.reshape(10000, 3, 32, 32).astype("float")
        else:
            X = X.reshape(50000, 3, 32, 32).astype("float")
        return X, Y

# 加载所有数据文件
def load_all_batches(batch_files, dataset='cifar10'):
    X_list, Y_list = [], []
    for file in batch_files:
        X, Y = load_cifar_batch(file, dataset)
        X_list.append(X)
        Y_list.append(Y)
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    return X_all, Y_all

# 获取图片和标签
def get_image_from_index(X_all, Y_all, index):
    img_array = X_all[index]
    label = Y_all[index]
    img = Image.fromarray(img_array.astype('uint8').transpose(1, 2, 0))
    return img, label

# 主函数
def main():
    dataset_choice = 'cifar100'  # 设置数据集选择为 'cifar10' 或 'cifar100'
    
    if dataset_choice == 'cifar10':
        batch_files = [
            './data/cifar-10-batches-py/data_batch_1',
            './data/cifar-10-batches-py/data_batch_2',
            './data/cifar-10-batches-py/data_batch_3',
            './data/cifar-10-batches-py/data_batch_4',
            './data/cifar-10-batches-py/data_batch_5'
        ]
    else:
        batch_files = [
            './data/cifar-100-python/train'
        ]
    
    X_all, Y_all = load_all_batches(batch_files, dataset=dataset_choice)

    # 设置要查看的图片索引
    index = 97
    if index < 0 or index >= len(Y_all):
        print("无效的索引，请输入有效的索引")
        return

    img, label = get_image_from_index(X_all, Y_all, index)

    # 显示图片和标签
    # img.show()
    print(f"Label: {label}")

if __name__ == "__main__":
    main()
