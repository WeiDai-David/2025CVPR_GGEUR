import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from scipy.spatial.distance import jensenshannon

# 计算KL散度，添加平滑项避免除以零
def kl_divergence(p, q):
    p = p + 1e-10  # 添加平滑项
    q = q + 1e-10  # 添加平滑项
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# 比较统计特征
def compare_statistics(real_features, generated_features):
    real_mean = np.mean(real_features, axis=0)
    generated_mean = np.mean(generated_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)
    generated_cov = np.cov(generated_features, rowvar=False)

    mean_diff = np.linalg.norm(real_mean - generated_mean)
    cov_diff = np.linalg.norm(real_cov - generated_cov)

    print(f"Mean difference: {mean_diff}")
    print(f"Covariance difference: {cov_diff}")

    real_hist, _ = np.histogram(real_features, bins=100, density=True)
    generated_hist, _ = np.histogram(generated_features, bins=100, density=True)
    kl_div = kl_divergence(real_hist, generated_hist)
    print(f"KL divergence: {kl_div}")

# 可视化比较
def visualize_features(real_features, generated_features):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)

    real_pca = pca.fit_transform(real_features)
    generated_pca = pca.fit_transform(generated_features)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real')
    plt.scatter(generated_pca[:, 0], generated_pca[:, 1], alpha=0.5, label='Generated')
    plt.legend()
    plt.title('PCA')

    real_tsne = tsne.fit_transform(real_features)
    generated_tsne = tsne.fit_transform(generated_features)

    plt.subplot(1, 2, 2)
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='Real')
    plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], alpha=0.5, label='Generated')
    plt.legend()
    plt.title('t-SNE')

    plt.show()

# 分类器评估
def evaluate_with_classifier(real_features, generated_features):
    labels = np.concatenate([np.zeros(len(real_features)), np.ones(len(generated_features))])
    features = np.concatenate([real_features, generated_features])

    classifier = SVC()
    classifier.fit(features, labels)
    predictions = classifier.predict(features)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    print(f"Classifier accuracy: {accuracy}")
    print(f"F1 score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

# 生成样本质量评估
def evaluate_generated_samples(real_features, generated_features):
    print("Comparing statistics...")
    compare_statistics(real_features, generated_features)

    print("Visualizing features...")
    visualize_features(real_features, generated_features)

    print("Evaluating with classifier...")
    evaluate_with_classifier(real_features, generated_features)


# 示例使用
real_features = np.load("F:/FederalLearning/MOON-main/David/CIFAR-10/features/initial/alpha=0.1_class_0_client_0/final_embeddings.npy")
generated_features = np.load("F:/FederalLearning/MOON-main/David/CIFAR-10/features/complete/alpha=0.1_class_0_client_1/final_embeddings_filled.npy")
evaluate_generated_samples(real_features, generated_features)
