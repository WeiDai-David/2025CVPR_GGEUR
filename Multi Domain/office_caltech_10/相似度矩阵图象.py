import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 从txt文件中加载数据
data = np.loadtxt('./output_similarity_results/5/amazon_vs_webcam_similarity_matrix.txt')  # 确保文件路径正确

# 生成热力图并保存
def save_heatmap(similarity_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# 使用加载的数据生成热力图
save_heatmap(data, "Similarity Matrix Heatmap", "./output_similarity_results/5/amazon_vs_webcam_heatmap_output.png")
