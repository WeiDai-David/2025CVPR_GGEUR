import os
import numpy as np

# 保存所有域的类分配数量到一个文件
def save_class_allocation_combined(domains, output_dir='./原始-output_indices'):
    combined_allocation = []

    # 遍历每个域
    for domain_name in domains:
        domain_output_dir = os.path.join(output_dir, domain_name)
        class_indices_path = os.path.join(domain_output_dir, 'class_indices.npy')
        client_indices_path = os.path.join(domain_output_dir, 'client_client_indices.npy')

        # 确保文件存在
        if not os.path.exists(class_indices_path) or not os.path.exists(client_indices_path):
            print(f"文件缺失: {class_indices_path} 或 {client_indices_path}")
            continue

        # 加载npy文件
        class_indices = np.load(class_indices_path, allow_pickle=True).item()
        client_indices = np.load(client_indices_path)

        # 初始化当前域的类分配
        domain_class_allocation = {class_label: 0 for class_label in class_indices.keys()}

        # 统计每个类的样本数量
        for idx in client_indices:
            for class_label, indices in class_indices.items():
                if idx in indices:
                    domain_class_allocation[class_label] += 1
                    break

        # 格式化当前域的类分配信息
        allocation_str = f"{domain_name}[" + ",".join(f"{class_label}:{count}" for class_label, count in domain_class_allocation.items()) + "]"
        combined_allocation.append(allocation_str)

    # 保存所有域的类分配信息到一个txt文件
    combined_txt_filename = os.path.join(output_dir, 'combined_class_allocation.txt')
    with open(combined_txt_filename, 'w') as f:
        for allocation in combined_allocation:
            f.write(f"{allocation}\n")
    print(f"已保存所有域的类分配数量到 {combined_txt_filename}")

# 主函数：处理所有域
def main():
    domains = ['Art', 'Clipart', 'Product', 'Real_World']
    save_class_allocation_combined(domains)

if __name__ == '__main__':
    main()
