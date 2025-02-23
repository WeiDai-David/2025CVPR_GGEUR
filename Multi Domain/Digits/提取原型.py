import os
import numpy as np

# 定义计算原型的函数
def compute_prototype(features):
    """
    根据特征计算当前类的原型。
    :param features: 属于某个类的特征 numpy 数组 (N, feature_dim)
    :return: 该类的原型，numpy 数组
    """
    return np.mean(features, axis=0)  # 计算平均值作为原型

# 处理函数，用于加载原始特征并计算原型
def process_prototypes(dataset_name, client_id, base_dir, output_dir, num_classes=10):
    """
    计算每个客户端每个类的原型并保存到文件。
    :param dataset_name: 数据集名称
    :param client_id: 客户端编号
    :param base_dir: 原始特征和标签文件的目录
    :param output_dir: 输出目录，用于保存原型文件
    :param num_classes: 类别数，默认是 10
    """
    for class_label in range(num_classes):
        # 加载原始特征和标签文件
        feature_file = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_label}_original_features.npy')
        label_file = os.path.join(base_dir, dataset_name, f'client_{client_id}_class_{class_label}_labels.npy')

        if os.path.exists(feature_file) and os.path.exists(label_file):
            features = np.load(feature_file)

            # 计算原型
            prototype = compute_prototype(features)

            # 保存每个类的原型到单独的文件
            prototype_file = os.path.join(output_dir, f'client_{client_id}_class_{class_label}_prototype.npy')
            np.save(prototype_file, prototype)
            print(f"已保存客户端 {client_id} 类别 {class_label} 的原型到 {prototype_file}")
        else:
            print(f"警告：客户端 {client_id} 类别 {class_label} 的特征或标签文件不存在，跳过...")

# 解析 dataset_report.txt 文件，确定每个数据集对应的客户端
def parse_dataset_report(report_file):
    dataset_clients = {}
    with open(report_file, 'r') as f:
        lines = f.readlines()
        current_dataset = None
        for line in lines:
            if "数据集大小" in line:
                current_dataset = line.split()[0]
                dataset_clients[current_dataset] = []
            elif "客户端" in line:
                client_id = int(line.split()[1])
                dataset_clients[current_dataset].append(client_id)
    return dataset_clients

# 主函数
def main(datasets, base_dir, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)

    # 解析 `dataset_report.txt` 文件，确定每个数据集对应的客户端
    dataset_clients = parse_dataset_report('./output_indices/dataset_report.txt')
    print(f"解析的客户端数据集映射: {dataset_clients}")

    for dataset_name in datasets:
        if dataset_name in dataset_clients:
            clients = dataset_clients[dataset_name]
            print(f"正在处理 {dataset_name} 数据集的客户端: {clients}")
            for client_id in clients:
                output_dir = os.path.join(output_base_dir, f'{dataset_name}')
                os.makedirs(output_dir, exist_ok=True)

                print(f"处理客户端 {client_id} 的原型")
                process_prototypes(dataset_name, client_id, base_dir, output_dir)
        else:
            print(f"警告：数据集 {dataset_name} 不存在客户端映射，跳过...")

if __name__ == "__main__":
    datasets = ['mnist', 'usps', 'svhn', 'syn']  # 要处理的数据集
    base_dir = './clip_features'  # 原始特征文件所在的目录
    output_base_dir = './clip_prototypes'  # 输出原型文件的目录
    main(datasets, base_dir, output_base_dir)
