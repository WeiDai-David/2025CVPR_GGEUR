import re

def extract_indices_from_text(text):
    """从字符串中提取索引，将其转换为整数列表"""
    # 使用正则表达式提取所有数字
    return list(map(int, re.findall(r'\d+', text)))

def load_indices_from_file(file_path):
    """从文件加载索引"""
    with open(file_path, 'r') as file:
        content = file.read()
    return extract_indices_from_text(content)

def check_subset_from_files(file1, file2):
    """检查第一个txt文件中的索引是否是第二个txt文件中的子集"""
    indices1 = load_indices_from_file(file1)
    indices2 = load_indices_from_file(file2)

    # 查找第一个文件的索引是否是第二个文件的子集
    is_subset = set(indices1).issubset(indices2)
    
    if is_subset:
        print(f"文件 {file1} 中的所有索引都是文件 {file2} 的子集。")
    else:
        # 如果不是子集，列出不在第二个文件中的索引
        missing_indices = set(indices1) - set(indices2)
        print(f"文件 {file1} 中的部分索引不在文件 {file2} 中。缺失的索引有: {sorted(list(missing_indices))}")

# 示例用法
file1 = './output_indices/dslr/client_3_indices.txt'  # 替换为第一个txt文件路径
file2 = './output_indices/dslr/train_train_indices.txt'  # 替换为第二个txt文件路径

check_subset_from_files(file1, file2)
