import matplotlib.pyplot as plt
import numpy as np

# Assume the number of samples is 5000, and each sample is a (1, 10) vector.
data_matrix1 = np.random.rand(5000, 10)
data_matrix2 = np.random.rand(5000, 10)

# Calculate the covariance matrix
covariance_matrix1 = np.cov(data_matrix1, rowvar=False)

# Perform eigenvalue decomposition on the covariance matrix
eigenvalues1, eigenvectors1 = np.linalg.eigh(covariance_matrix1)

# Sort the eigenvalues
sorted_indices = np.argsort(eigenvalues1)[::-1]
eigenvalues1 = eigenvalues1[sorted_indices]
eigenvectors1 = eigenvectors1[:, sorted_indices]


# Calculate the covariance matrix
covariance_matrix2 = np.cov(data_matrix2, rowvar=False)

# Perform eigenvalue decomposition on the covariance matrix
eigenvalues2, eigenvectors2 = np.linalg.eigh(covariance_matrix2)

# Sort the eigenvalues
sorted_indices = np.argsort(eigenvalues2)[::-1]
eigenvalues2 = eigenvalues2[sorted_indices]
eigenvectors2 = eigenvectors2[:, sorted_indices]

similarity = 0
for i in range(len(eigenvectors2)):
    similarity += np.abs(np.dot(eigenvectors1[:,i].T,eigenvectors2[:,i]))

print("Similarity of the geometric shapes of the two perceptual manifolds:", similarity)