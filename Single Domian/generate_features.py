import numpy as np
import pickle
from tqdm import tqdm


def generate_new_samples(class1_features, class2_covariance_matrix, num_generated):
    new_features_list = []
    eigenvalues, eigenvectors = np.linalg.eigh(class2_covariance_matrix)
    eigenvalues = eigenvalues

    B = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    for feature in class1_features:

        new_features = np.random.multivariate_normal(feature, B, num_generated)
        new_features_list.append(new_features)

    new_features_array = np.vstack(new_features_list)
    return new_features_array



dict_file_path = r"../cifar10_r200_TO5000_features.pkl"


with open("simi_cov_matrix.pkl", 'rb') as f:
    cov_dict = pickle.load(f)



with open("../cifar10_r200_train_features.pkl", "rb") as f:
    cifar10_r200_train_features = pickle.load(f)



cifar10_r200_TO5000_features = {}

cifar10_r200_TO5000_features[0] = cifar10_r200_train_features[0]


for i in tqdm(range(1, 10), desc='Generating and Augmenting Samples'):
    new_samples = generate_new_samples(cifar10_r200_train_features[i], cov_dict[i], num_generated=200)
    num_samples = 5000 - len(cifar10_r200_train_features[i])
    random_selected_samples_indices = np.random.choice(new_samples.shape[0], num_samples, replace=False)
    random_selected_samples = new_samples[random_selected_samples_indices]
    cifar10_r200_TO5000_features[i] = np.vstack((cifar10_r200_train_features[i], random_selected_samples))

with open(dict_file_path, "wb") as f:
    pickle.dump(cifar10_r200_TO5000_features, f)




