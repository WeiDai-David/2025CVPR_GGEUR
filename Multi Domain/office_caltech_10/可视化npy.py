import numpy as np
import os

# 修改此路径为你要查看的 .npy 文件路径
file_path = './clip_office_caltech_train_features/dslr/client_3_class_0_labels.npy'

def view_npy(file_path):
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return
    
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"文件 {file_path} 的内容:")
        print(data)
    except Exception as e:
        print(f"读取 {file_path} 文件时出错: {e}")

if __name__ == "__main__":
    view_npy(file_path)
