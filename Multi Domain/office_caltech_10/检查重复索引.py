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

def check_duplicates_from_files(file1, file2):
    """检查两个txt文件中的索引是否有重复"""
    indices1 = load_indices_from_file(file1)
    indices2 = load_indices_from_file(file2)

    # 查找重复的索引
    duplicates = set(indices1).intersection(indices2)
    
    if duplicates:
        print(f"发现 {len(duplicates)} 个重复的索引： {sorted(list(duplicates))}")
    else:
        print("没有重复的索引。")

# 示例用法
file1 = './output_indices/dslr/train_train_indices.txt'  # 替换为第一个txt文件路径
# file2 = './Train0.3-Test0.2/output_indices/cartoon/test_test_indices.txt'  # 替换为第二个txt文件路径
file2 = './output_indices/dslr/test_test_indices.txt'

check_duplicates_from_files(file1, file2)
