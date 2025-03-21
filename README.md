<!-- # 2025CVPR_GGEUR
联邦学习中几何引导的本地全局分布对齐

摘要：联邦学习中的数据异质性，其特征在于局部和全局分布之间的显著不一致，导致局部优化方向不同，并阻碍全局模型训练。现有的研究主要集中在优化本地更新或全局聚合，但这些间接方法在处理高度异构的数据分布时表现出不稳定性，特别是在标签偏斜和域偏斜共存的情况下。为了解决这个问题，我们提出了一种几何引导的数据增强方法，该方法侧重于在局部模拟全局嵌入分布。我们首先引入了嵌入分布几何形状的概念，并在隐私约束下解决了获取全局几何形状的挑战。随后，我们提出了GGEUR，它利用全局几何形状来引导新样本的生成，从而实现对理想全局分布更好的近似。在单域的情况下，我们通过基于全局几何形状的样本增强来提高模型泛化能力；在多域的情况下，我们进一步采用类原型来模拟跨域的全局分布。大量的实验结果表明，我们的方法在处理高度异构的数据时显著提升了性能，尤其是在标签倾斜，域倾斜，和二者共存的情况下。

关键词：联邦学习、数据异质、标签偏移、域泛化、感知流形 -->

# Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning

> **Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning**  
> Yanbiao Ma*, Wei Dai*, Wenke Huang†, Jiayi Chen  
> Accepted to CVPR 2025 [🔗Link](https://arxiv.org/pdf/2503.06457)

<!-- **项目整理中 待完成的工作 7/3/2025**：

<<<<<<< HEAD
(1):英文版本-done <br>
(2):各个子工程脚本名的逻辑重构-done <br>
(3):子工程内脚本的逻辑关系图 子工程间的逻辑关系图<br>
(4):站在模型视野(流形空间)的角度将我们的思想以图的方式剖析<br>
(5):更优美更逻辑的 md 表述-done <br> -->
=======
(1):英文版本<br>
(2):各个子工程脚本名的逻辑重构<br>
(3):子工程内脚本的逻辑关系图 子工程间的逻辑关系图<br>
(4):站在模型视野(流形空间)的角度将我们的思想以图的方式剖析<br>
(5):更优美更有逻辑的 md 表述<br> -->
>>>>>>> ea46942d7ea01c53b824c0bad7df298a61034415

# 📝 Abstract

Data heterogeneity in federated learning, characterized by a significant misalignment between local and global distributions, leads to divergent local optimization directions and hinders global model training.Existing studies mainly focus on optimizing local updates or global aggregation, but these indirect approaches demonstrate instability when handling highly heterogeneous data distributions, especially in scenarios where label skew and domain skew coexist.To address this, we propose a geometry-guided data generation method that centers on simulating the global embedding distribution locally. We first introduce the concept of the geometric shape of an embedding distribution and then address the challenge of obtaining global geometric shapes under privacy constraints. Subsequently, we propose GGEUR, which leverages global geometric shapes to guide the generation of new samples, enabling a closer approximation to the ideal global distribution.In single-domain scenarios, we augment samples based on global geometric shapes to enhance model generalization;in multi-domain scenarios, we further employ class prototypes to simulate the global distribution across domains.Extensive experimental results demonstrate that our method significantly enhances the performance of existing approaches in handling highly heterogeneous data, including scenarios with label skew, domain skew, and their coexistence.

<!-- --- -->

<!-- ## 🔑 Key words

**Federated Learning**, **Data Heterogeneity**, **Domain Generalization**, **Perceptual Manifold** -->

---

# 📂 New Dataset: Office-Home-LDS

**Dataset & Constructor:** 👉 [📥 Huggingface](https://huggingface.co/datasets/WeiDai-David/Office-Home-LDS)

The dataset is organized as follows:

```text
Office-Home-LDS/
├── data/
│   └── Office-Home.zip        #  Original raw dataset (compressed)
├── new_dataset/               #  Processed datasets based on different settings
│   ├── Office-Home-0.1.zip    #  Split with Dirichlet α = 0.1 (compressed)
│   ├── Office-Home-0.5.zip    #  Split with Dirichlet α = 0.5 (compressed)
│   └── Office-Home-0.05.zip   #  Split with Dirichlet α = 0.05 (compressed)
├── Dataset-Office-Home-LDS.py #  Python script for processing and splitting original raw dataset
└── README.md                  #  Project documentation
```

---

<!-- # ⚙️ Engineering -->

# 🌐 Environment

## Provide two options

**(1): Download dependency**

```bash
conda create -n GGEUR python=3.9
conda activate GGEUR
pip install -r requirements.txt
```

**(2): Download environment (prefer)**

```bash
conda env create -f environment.yaml
```

---

# 🚀 Single Domain

---

## 🗂️ 1. Dataset Partitioning

### CIFAR 10 & 100 dataset

#### Dataset index parsing:

> The CIFAR dataset is sorted by subsets of the same class, so the indexing process is directly arranged by the number of classes.

#### Optional Parameters

| Parameter          | Type    | Description             |
| ------------------ | ------- | ----------------------- |
| `num_clients`      | integer | Number of clients       |
| `alpha`            | float   | Dirichlet coefficient   |
| `min_require_size` | integer | Minimum allocation size |

#### Run Script

```bash
python data_distribution_CIFAR-10.py
python data_distribution_CIFAR-100.py
python data_batch2index_images.py  # Check the partitioning
```

#### Output

```bash
{dataset_name}/context/alpha={alpha}_class_counts.txt         #  Total count of each class after partitioning (for validation)
{dataset_name}/context/alpha={alpha}_class_indices.txt        #  Class indices based on partitioning
{dataset_name}/context/alpha={alpha}_client_indices.txt       #  Data indices assigned to each client
{dataset_name}/context/alpha={alpha}_client_class_distribution.txt #  Class distribution across clients
{dataset_name}/images/alpha={alpha}_label_distribution_heatmap.png #  Heatmap showing the number of each class assigned to each client
```

---

### TinyImageNet dataset

#### Dataset Structure

```bash
📌 The TinyImageNet dataset is derived from 200 classes of ImageNet.
⚠️ The class order is not linear, so reconstruction is required.

TinyImageNet Structure:
│
├── train/
│   └── 🗂️ n01443537/   # Class folder
│       └──  images/  # Training images for the class
│           └──  image_001.JPEG
│           └──  image_002.JPEG
│           └── ...
├── val/
│   ├──  images/       # Validation images
│   └──  val_annotations.txt  # Mapping between images and class IDs
│
├── test/
│    └──  images/       # Test images
│
├── wnids.txt         # Class ID list
│
└── words.txt         # Class label list
```

👉 **Since the test set has no labels**, the validation set will be used as the test set.  
👉 The training set follows the same folder structure as ImageNet.  
👉 Validation images are stored in `val/images`, and the label mapping is in `val/val_annotations.txt`.
👉 The script `Reorganized_TinyImageNet_Val.py` aligns the validation set structure with the training set:

```bash
│
├── train/
│   └── 🗂️ n01443537/   # Class folder
│       └──  images/  # Training images for the class
│           └──  image_001.JPEG
│           └──  image_002.JPEG
│           └── ...
├── new_val/
│   └── 🗂️ n01443537/   # Class folder
│       └──  images/  # Validation images for the class
│           └──  image_001.JPEG
│           └──  image_002.JPEG
│           └── ...
```

💡 **After this:**  
Label IDs (`n + 8 digits`) will be indexed.For consistency, the `n` prefix will be removed, and IDs will be sorted to define the class order.

📌 This is handled by:

- 📝 `data_distribution_TinyImageNet.py` (training set partitioning)
- 📝 `TinyImageNet_Val.py` (validation set partitioning)
  At this point, we have completed the reconstruction of the TinyImageNet dataset, laying the groundwork for the subsequent indexing and partitioning tasks.

#### Parameters

| Parameter          | Type    | Description             |
| ------------------ | ------- | ----------------------- |
| `num_clients`      | integer | Number of clients       |
| `alpha`            | float   | Dirichlet coefficient   |
| `min_require_size` | integer | Minimum allocation size |

#### Run Scripts

```bash
python reorganized_TinyImageNet_val.py         # Reorganize validation set
python TinyImageNet_val_index.py               # Convert validation set indices
python data_distribution_TinyImageNet.py       #  Partition training set
python TinyImageNet_val_index_tag_img_matching_test.py  # Validate processed validation set
python TinyImageNet_val_features.py            # Extract validation set features
```

#### Output

```bash
{dataset_name}/context/alpha={alpha}_class_counts.txt         # Total count of each class after partitioning (for validation)
{dataset_name}/context/alpha={alpha}_class_indices.txt        # Class indices based on partitioning
{dataset_name}/context/alpha={alpha}_client_indices.txt       # Data indices assigned to each client
{dataset_name}/context/alpha={alpha}_client_class_distribution.txt # Class distribution across clients
{dataset_name}/images/alpha={alpha}_label_distribution_heatmap.png # Heatmap showing the number of each class assigned to each client
{dataset_name}/context/class_map.txt                          # Mapping between class index and class ID (training set)
{dataset_name}/val_context/class_map.txt                      # Mapping between class index and class ID (validation set)
{dataset_name}/val_context/val_indices.npy                    # Validation set indices
{dataset_name}/val_context/val_labels.npy                     # Labels corresponding to validation set indices
{dataset_name}/class_{class_label}_val_indices.npy            # Indices of each class in validation set
```

---

## ✅ 2. Cross Indexing

We have obtained the data indexes for the CIFAR-10, CIFAR-100, and TinyImageNet datasets, including the data indexes for each client and each class.

By performing cross-indexing between the two, we can generate the data indexes for each class under each client.

Taking CIFAR-10 as an example: **10 clients × 10 classes = 100 cross-index files**

### Parameters

| Parameter | Type   | Description                                      |
| --------- | ------ | ------------------------------------------------ |
| `dataset` | string | Dataset name (CIFAR-10, CIFAR-100, TinyImageNet) |
| `alpha`   | float  | Dirichlet coefficient                            |

### Run Script

```bash
python client_class_cross_index.py
```

### Output

```bash
{dataset_name}/client_class_indices/alpha={alpha}_{dataset}_client_{client_id}_class_{class_id}_indices.npy
```

---

## ✅ 3. Feature Extraction

We have obtained the index files of each class for each client in the three datasets.  
Using **CLIP** as the backbone, we extract features for each index file and generate the corresponding feature and label files.

### Parameters

| Parameter | Type   | Description                                      |
| --------- | ------ | ------------------------------------------------ |
| `dataset` | string | Dataset name (CIFAR-10, CIFAR-100, TinyImageNet) |
| `alpha`   | float  | Dirichlet coefficient                            |

### Run Script

```bash
python client_class_clip_features2tensor.py
```

### Output

```bash
{dataset_name}/features/initial/alpha={alpha}_class_{class_idx}_client_{client_idx}/final_embeddings.npy
{dataset_name}/features/initial/alpha={alpha}_class_{class_idx}_client_{client_idx}/labels.npy
```

---

## ✅ 4. Global Distribution

### 🔸 4.1 Global Distribution of Client Composition

The file `alpha={alpha}_client_class_distribution.txt` generated in Section 1 contains the class distribution for each client.  
Clients are sorted based on the number of classes, and a set of clients approximating the global distribution is determined according to a threshold ratio.

#### Parameters

| Parameter   | Type   | Description                                      |
| ----------- | ------ | ------------------------------------------------ |
| `threshold` | float  | Threshold                                        |
| `dataset`   | string | Dataset name (CIFAR-10, CIFAR-100, TinyImageNet) |
| `alpha`     | float  | Dirichlet coefficient                            |

#### Run Script

```bash
python client-guided_set.py
```

#### Output

```bash
{dataset_name}/context/alpha={alpha}_selected_clients_for_each_class.txt
```

---

### 🔸 4.2 Client Set Features (Optional)

In Section 3, we obtained feature files for each class for each client.  
In Section 4.1, we obtained the client sets that approximate the global distribution for each class.  
We now extract the corresponding feature files from the former using the latter.

#### Parameters

| Parameter | Type   | Description                                      |
| --------- | ------ | ------------------------------------------------ |
| `dataset` | string | Dataset name (CIFAR-10, CIFAR-100, TinyImageNet) |
| `alpha`   | float  | Dirichlet coefficient                            |

#### Run Script

```bash
python client-guided_clip_tensor.py
```

---

### 🔸 4.3 Representation of Global Distribution

If the client set contains only one client, the covariance matrix is calculated directly from the feature matrix.  
If the client set contains multiple clients, the covariance matrix is calculated separately for each client’s feature matrix, and the resulting covariance matrices are aggregated to form the final aggregated covariance matrix.

#### Parameters

| Parameter | Type   | Description                                      |
| --------- | ------ | ------------------------------------------------ |
| `dataset` | string | Dataset name (CIFAR-10, CIFAR-100, TinyImageNet) |
| `alpha`   | float  | Dirichlet coefficient                            |

#### Run Script

```bash
python clip_tensor2aggregate_covariance_matrix.py
```

#### Output

```bash
{dataset_name}/features/alpha={alpha}_cov_matrix/class_{idx}_cov_matrix.npy
```

---

## ✅ 5. Geometry-Guided Data Augmentation

Now that the global distribution framework (covariance matrix) has been obtained, the covariance matrix is decomposed to derive eigenvalues and eigenvectors (geometric directions).  
These geometric directions are used to guide the augmentation of raw samples from the client set.

### Parameters

| Parameter | Type   | Description                                      |
| --------- | ------ | ------------------------------------------------ |
| `dataset` | string | Dataset name (CIFAR-10, CIFAR-100, TinyImageNet) |
| `alpha`   | float  | Dirichlet coefficient                            |

### Run Script

```bash
python cov_matrix_generate_features.py
```

### Output

```bash
{dataset_name}/features/alpha={alpha}_complete/final_embeddings_filled.npy
{dataset_name}/features/alpha={alpha}_complete/labels_filled.npy
```

---

## ✅ 6. Single Client Training

Under a non-federated learning architecture, train both the original and augmented samples locally and compare the performance differences.

### Parameters

| Parameter       | Type    | Description               |
| --------------- | ------- | ------------------------- |
| `alpha`         | float   | Dirichlet coefficient     |
| `client_idx`    | integer | Client ID                 |
| `batch_size`    | integer | Batch size                |
| `learning_rate` | float   | Learning rate             |
| `num_epochs`    | integer | Number of training epochs |

### Run Script

```bash
python MLP_10.py   # Train on CIFAR-10 dataset
python MLP_100.py  # Train on CIFAR-100 dataset
python MLP_200.py  # Train on TinyImageNet dataset
```

---

## ✅ 7. Federated Architecture FedAvg Training

Under a basic federated learning architecture (FedAvg with simple averaging), train models using both the original and augmented samples and compare the performance differences.

### Parameters

| Parameter              | Type    | Description                     |
| ---------------------- | ------- | ------------------------------- |
| `alpha`                | float   | Dirichlet coefficient           |
| `communication_rounds` | integer | Number of communication rounds  |
| `local_epochs`         | integer | Number of local training epochs |
| `client_count`         | integer | Number of clients               |
| `batch_size`           | integer | Batch size                      |
| `learning_rate`        | float   | Learning rate                   |

### Run Script

```bash
python FL_MLP_10.py   # Train on CIFAR-10 dataset
python FL_MLP_100.py  # Train on CIFAR-100 dataset
python FL_MLP_200.py  # Train on TinyImageNet dataset
```

---

# 🚀 Cross-Domain Scenarios

We conducted experiments on the **Digits**, **PACS**, and **Office-Caltech-10** datasets, and proposed a new dataset called **Office-Home-LDS**.<br>
Following standard cross-domain work:

- Each domain is assigned one client.
- Data is randomly partitioned based on ratio used in prior work.
- For datasets without explicit training and test splits, we follow the ratio used in prior work.

---

## 🏷️ Digits

### 📌1. Dataset Overview

The **Digits** dataset contains data from four different domains, representing digits from 0 to 9:

- **MNIST** – 🖋️ Handwritten digits from different individuals (0-9)
- **SVHN** – 🏠 Street View House Numbers (0-9)
- **USPS** – 📬 Postal Service digits (0-9)
- **SynthDigits** – 🎨 Synthetic RGB digits (0-9)

---

### 📂 2. Dataset Partitioning

Following the standard settings for federated cross-domain work:

- Each domain is assigned one client.
- Data is randomly partitioned based on ratio used in prior work.
- For datasets without explicit training and test splits, we follow the ratio used in prior work.

#### Run Script

```bash
python data_distribution_digits.py
```

#### Output

```bash
output_indices/{domain_name}/client_{client_id}_indices.npy         # Indices assigned to the client within the domain
output_indices/{domain_name}/combined_class_indices.npy             # Combined class indices within the domain
output_indices/client_combined_class_distribution.txt              # Class distribution per client within each domain
output_indices/dataset_report.txt                                  # Total samples assigned to each client (for validation)
```

---

### ✅ 3. Cross Indexing

We have obtained client indices and class indices for each domain.  
By performing cross-indexing, we can generate class-specific indices for each client.

#### Run Script

```bash
python client_class_cross_index.py
```

#### Output

```bash
output_client_class_indices/{domain_name}/client_{client_id}_class_{0~9}_indices.npy
```

---

### ✅ 4. Training Set Feature Extraction

Using **CLIP** as the backbone, we extract features and labels for each client-class index file.

#### Parameters

| Parameter     | Type   | Description                                  |
| ------------- | ------ | -------------------------------------------- |
| `datasets`    | string | Domain name (MNIST, SVHN, USPS, SynthDigits) |
| `report_file` | string | Number of communication rounds               |

#### Run Script

```bash
python train_client_class_clip_features2tensor.py
```

#### Output

```bash
clip_features/{domain_name}/client_{client_id}_class_{0~9}_original_features.npy
clip_features/{domain_name}/client_{client_id}_class_{0~9}_labels.npy
```

---

### ✅ 5. Test Set Feature Extraction

We extract features and labels for the test set using CLIP as the backbone.

#### Parameters

| Parameter  | Type   | Description                                  |
| ---------- | ------ | -------------------------------------------- |
| `datasets` | string | Domain name (MNIST, SVHN, USPS, SynthDigits) |

#### Run Script

```bash
python test_clip_features2tensor.py
```

#### Output

```bash
clip_test_features/{domain_name}/{domain_name}_test_features.npy
clip_test_features/{domain_name}/{domain_name}_test_labels.npy
```

---

### ✅ 6. Prototype Extraction

Using the client-class index files, we extract prototypes for each class.

#### Run Script

```bash
python prototype_clip_features2tensor.py
```

#### Output

```bash
clip_prototypes/{domain_name}/client_{client_id}_class_{0~9}_prototype.npy
```

---

### ✅ 7. Geometric Direction

From the **perspective** of the **manifold space**, cross-domain differences are caused by **shifts in class distribution**, but the **geometric structure remains unchanged**.  
Thus, we can use the combined features from multiple domains to represent the geometric structure.

#### Parameters

| Parameter     | Type   | Description                     |
| ------------- | ------ | ------------------------------- |
| `report_file` | string | File mapping domains to clients |

#### Run Script

```bash
python clip_tensor2aggregate_covariance_matrix.py
```

#### Output

```bash
cov_matrix_output/class_{0~9}_cov_matrix.npy
```

---

### ✅ 8. Geometry-Guided Data Augmentation

We now have the geometric structure and class prototypes for multiple domains.  
For data augmentation:

- For samples within the same domain, apply domain-specific augmentation.
- For samples outside the domain, augment based on class prototypes and geometric structure.

This allows the client to learn cross-domain features even from its own perspective, thereby effectively mitigating the bias caused by domain shift.

#### Run Script

```bash
python prototype_cov_matrix_generate_features.py
```

#### Output

```bash
argumented_clip_features/{domain_name}/client_{client_id}_class_{0~9}/final_embeddings_filled.npy
argumented_clip_features/{domain_name}/client_{client_id}_class_{0~9}/labels_filled.npy
```

---

### ✅ 9. Training Under Different Federated Architectures

We train both the original and augmented models under different federated architectures to compare performance:

#### Run Script

```bash
python FedAvg.py
python FedAvg_GGEUR.py
python FedNTD.py
python FedNTD_GGEUR.py
python FedOpt.py
python FedOpt_GGEUR.py
python FedProx.py
python FedProx_GGEUR.py
python MOON.py
python MOON_GGEUR.py
python FedDyn.py
python FedDyn_GGEUR.py
python FedProto.py
python FedProto_GGEUR.py
python SCAFFOLD.py
python SCAFFOLD_GGEUR.py
```

---

## 🌍 PACS and Office-Caltech-10

PACS and Office-Caltech-10 are small-sample datasets with fewer classification tasks:

- ✅ Fewer classification tasks – Low classification difficulty.
- ✅ High inter-class variation – Easier to distinguish between classes.
- ✅ Similar training and test sets – Improves accuracy implicitly.
- ✅ Balanced intra-domain classes – Following a proportional random split.

With the development of self-supervised learning, the challenges posed by cross-domain datasets like PACS and Office_Caltech_10 have been significantly weakened when using backbones such as CLIP and DINO.

Therefore, there is an urgent need for a more realistic and challenging dataset that better reflects real-world scenarios. To address this, we propose **Office-Home-LDS** — a dataset built upon the Office-Home dataset (65 classes) that incorporates both cross-domain(**domain skew**) and data heterogeneity(**label skew**). This clearly aligns better with real-world conditions.

---

## 🏷️ PACS

### 📌 1. Dataset Overview

The PACS dataset contains data from four different domains, with seven categories:  
**Dog**, **Elephant**, **Giraffe**, **Guitar**, **Horse**, **House**, **Person**

- **P** (Photo) – 📷 Real-world photos
- **A** (Art Painting) – 🎨 Artistic paintings
- **C** (Cartoon) – 🐾 Cartoon images
- **S** (Sketch) – ✍️ Sketches

---

### 📂 2. Dataset Partitioning

Following standard cross-domain work:

- Each domain is assigned one client.
- Data is randomly partitioned based on ratio used in prior work.
- For datasets without explicit training and test splits, we follow the ratio used in prior work.

#### Run Script

```bash
python data_distribution_digits.py
```

#### Output

```bash
./output_indices/{domain_name}/train_train_indices.npy          # Training set indices for the domain
./output_indices/{domain_name}/test_test_indices.npy            # Test set indices for the domain
./output_indices/{domain_name}/client_{client_id}_indices.npy   # Client-assigned indices for the domain
./output_indices/{domain_name}/class_indices.npy                # Combined class indices within the domain
./output_indices/client_combined_class_distribution.txt         # Class distribution per client within each domain
```

---

### ✅ 3. Cross Indexing

We have obtained the client indices and class indices for each domain.  
By performing cross-indexing, we can generate class-specific indices for each client.

#### Run Script

```bash
python client_class_cross_index.py
```

#### Output

```bash
output_client_class_indices/{domain_name}/client_{client_id}_class_{0~6}_indices.npy
```

---

### ✅ 4. Training Set Feature Extraction

We have obtained class-specific index files for each client in the four domains.  
Using **CLIP** as the backbone, we extract features for each index file and generate the corresponding feature and label files.

#### Run Script

```bash
python train_client_class_clip_features2tensor.py
```

#### Output

```bash
clip_pacs_train_features/{domain_name}/client_{client_id}_class_{0~6}_original_features.npy
clip_pacs_train_features/{domain_name}/client_{client_id}_class_{0~6}_labels.npy
```

---

### ✅ 5. Test Set Feature Extraction

We extract features and labels for the test set using CLIP as the backbone.

#### Run Script

```bash
python test_clip_features2tensor.py
```

#### Output

```bash
clip_test_features/{domain_name}/{domain_name}_test_features.npy
clip_test_features/{domain_name}/{domain_name}_test_labels.npy
```

---

### ✅ 6. Training Under Different Federated Architectures

We train both the original and augmented models under different federated architectures to compare performance:

#### Run Script

```bash
python FedAvg.py
python FedNTD.py
python FedOpt.py
python FedProx.py
python MOON.py
python FedDyn.py
python FedProto.py
python SCAFFOLD.py
```

---

## 🏷️ Office-Caltech-10

### 📌 1. Dataset Overview

The Office-Caltech-10 dataset contains data from four different domains, representing 10 categories:  
**Headphones**, **Keyboard**, **Laptop**, **Monitor**, **Mouse**, **Mug**, **Projector**, **Bike**

- **A** (Amazon) – 🛒 Product images from Amazon
- **W** (Webcam) – 📷 Images captured from a webcam
- **D** (DSLR) – 📸 Images captured from a DSLR camera
- **C** (Caltech-256) – 🎯 Selected images from the Caltech-256 dataset

---

### 📂 2. Dataset Partitioning

Following standard federated cross-domain work:

- Each domain is assigned one client.
- Data is randomly partitioned based on a predefined ratio.
- For datasets without explicit training and test splits, we follow the ratio used in prior work.

#### Run Script

```bash
python data_distribution_office_caltech_10.py
```

#### Output

```bash
output_indices/{domain_name}/train_train_indices.npy         # Training set indices for the domain
output_indices/{domain_name}/test_test_indices.npy           # Test set indices for the domain
output_indices/{domain_name}/client_{client_id}_indices.npy  # Client-assigned indices for the domain
output_indices/{domain_name}/class_indices.npy               # Combined class indices within the domain
output_indices/client_combined_class_distribution.txt        # Class distribution per client within each domain
```

---

### ✅ 3. Cross Indexing

We have obtained client indices and class indices for each domain.  
By performing cross-indexing, we can generate class-specific indices for each client.

#### Run Script

```bash
python client_class_cross_index.py
```

#### Output

```bash
output_client_class_indices/{domain_name}/client_{client_id}_class_{0~64}_indices.npy
```

---

### ✅ 4. Training Set Feature Extraction

We have obtained class-specific index files for each client in the four domains.  
Using **CLIP** as the backbone, we extract features for each index file and generate the corresponding feature and label files.

#### Run Script

```bash
python train_client_class_clip_features2tensor.py
```

#### Output

```bash
clip_office_home_train_features/{domain_name}/client_{client_id}_class_{0~64}_original_features.npy
clip_office_home_train_features/{domain_name}/client_{client_id}_class_{0~64}_labels.npy
```

---

### ✅ 5. Test Set Feature Extraction

We extract features and labels for the test set using CLIP as the backbone.

#### Run Script

```bash
python test_clip_features2tensor.py
```

#### Output

```bash
clip_test_features/{domain_name}/{domain_name}_test_features.npy
clip_test_features/{domain_name}/{domain_name}_test_labels.npy
```

---

### ✅ 6. Training Under Different Federated Architectures

We train both the original and augmented models under different federated architectures to compare performance:

##### Run Script

```bash
python FedAvg.py
python FedNTD.py
python FedOpt.py
python FedProx.py
python MOON.py
python FedDyn.py
python FedProto.py
python SCAFFOLD联.py
```

---

## 🏷️ Office-Home-LDS

### 📌 1. Dataset Overview

The **Office-Home-LDS** dataset contains data from four different domains, covering **65 categories**:

- **Art** – 🎨 Hand-drawn, sketch, or oil painting-style images
- **Clipart** – 🖼️ Cartoon and clipart images from the web
- **Product** – 🛒 Product or merchandise display images
- **Real World** – 🌍 Photographs captured from real-world scenarios

---

### 📂 2. Dataset Partitioning

Following standard federated cross-domain work:

- Each client is assigned one domain.
- Data is randomly partitioned based on the Dirichlet distribution within the training set.
- For datasets without explicit training and test splits, we follow the ratio used in prior work.

#### Run Script

```bash
python data_distribution_Office_Home_LDS.py
```

#### Output

```bash
./output_indices/{domain_name}/train_train_indices.npy         # Training set indices for the domain
./output_indices/{domain_name}/test_test_indices.npy           # Test set indices for the domain
./output_indices/{domain_name}/client_{client_id}_indices.npy  # Client-assigned indices for the domain
./output_indices/{domain_name}/class_indices.npy               # Combined class indices within the domain
./output_indices/client_combined_class_distribution.txt        # Class distribution per client within each domain
```

---

### ✅ 3. Cross Indexing

We have obtained client indices and class indices for each domain.  
By performing cross-indexing, we can generate class-specific indices for each client.

#### Run Script

```bash
python client_class_cross_index.py
```

#### Output

```bash
output_client_class_indices/{domain_name}/client_{client_id}_class_{0~64}_indices.npy
```

---

### ✅ 4. Training Set Feature Extraction

We have obtained class-specific index files for each client in the four domains.  
Using **CLIP** as the backbone, we extract features for each index file and generate the corresponding feature and label files.

#### Run Script

```bash
python train_client_class_clip_features2tensor.py
```

#### Output

```bash
clip_office_home_train_features/{domain_name}/client_{client_id}_class_{0~64}_original_features.npy
clip_office_home_train_features/{domain_name}/client_{client_id}_class_{0~64}_labels.npy
```

---

### ✅ 5. Test Set Feature Extraction

We extract features and labels for the test set using CLIP as the backbone.

#### Run Script

```bash
python test_clip_features2tensor.py
```

#### Output

```bash
clip_test_features/{domain_name}/{domain_name}_test_features.npy
clip_test_features/{domain_name}/{domain_name}_test_labels.npy
```

---

### ✅ 6. Prototype Extraction

Using the client-class index files obtained earlier, we extract class prototypes for each client.

#### Run Script

```bash
python prototype_clip_features2tensor.py
```

#### Output

```bash
./office_home_prototypes/{domain_name}/client_{client_id}_class_{0~64}_prototype.npy
```

---

### ✅ 7. Geometric Direction

From the perspective of the manifold space, cross-domain differences are caused by shifts in class distribution, but the geometric structure remains unchanged.  
Thus, we can use the combined features from multiple domains to represent the geometric structure.

#### Parameters

| Parameter     | Type   | Description                     |
| ------------- | ------ | ------------------------------- |
| `report_file` | string | File mapping domains to clients |

#### Run Script

```bash
python clip_tensor2aggregate_covariance_matrix.py
```

#### Output

```bash
cov_matrix_output/class_{0~64}_cov_matrix.npy
```

---

### ✅ 8. Geometry-Guided Data Augmentation

We now have the geometric direction and class prototypes for multiple domains.  
For data augmentation:

- For samples within the same domain, apply a single-domain augmentation strategy.
- For samples outside the domain, augment based on class prototypes and geometric directions.
- This allows the client to learn cross-domain features and mitigate domain shift effectively.

#### Run Script

```bash
python prototype_cov_matrix_generate_features.py
```

#### Output

```bash
argumented_clip_features/{domain_name}/client_{client_id}_class_{0~64}/final_embeddings_filled.npy
argumented_clip_features/{domain_name}/client_{client_id}_class_{0~64}/labels_filled.npy
```

---

### ✅ 9. Training Under Different Federated Architectures

We train both the original and augmented models under different federated architectures to compare performance:

#### Run Script

```bash
python FedAvg.py
python FedAvg_GGEUR.py
python FedNTD.py
python FedNTD_GGEUR.py
python FedOpt.py
python FedOpt_GGEUR.py
python FedProx.py
python FedProx_GGEUR.py
python MOON.py
python MOON_GGEUR.py
python FedDyn.py
python FedDyn_GGEUR.py
python FedProto.py
python FedProto_GGEUR.py
python SCAFFOLD.py
python SCAFFOLD_GGEUR.py
```

---

## Learning Trajectory (Updating...)

When I completed this project, I was a third-year undergraduate student. 🌿 I will share my learning trajectory and how to efficiently and comprehensively develop expertise in a specific field. 🌊 I believe that the most effective approach is to start by identifying high-quality review articles from top-tier journals. 📚 After forming a comprehensive understanding of the field, I recommend selecting detailed papers from the references cited in these outstanding reviews, focusing on those that align with the direction of our current work for in-depth study. 🔍 This process resembles a leaf with its veins hollowed out — our process of understanding is akin to a flood flowing through the leaf, with the central vein serving as the core from which knowledge selectively branches out in all directions. 🚀

- **2023Tpami** "Deep Long-Tailed Learning: A Survey" [Paper](https://arxiv.org/pdf/2304.00685)——Review on Long-Tailed Learning

- **2024Tpami** "Vision-Language Models for Vision Tasks: A Survey" [Paper](https://arxiv.org/pdf/2304.00685) & [Github](https://github.com/jingyi0000/VLM_survey)——Review on Vision-Language Large Models

- **2024Tpami** "Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark" [Paper](https://arxiv.org/pdf/2311.06750) & [Github](https://github.com/WenkeHuang/MarsFL)——Review on Federated Learning

- **2021CVPR** "Model-Contrastive Federated Learning" [Paper](https://arxiv.org/pdf/2103.16257) & [Github](https://github.com/QinbinLi/MOON)——MOON(Alignment of Local and Global Model Representations)

- **2022AAAI** "FedProto: Federated Prototype Learning across Heterogeneous Clients" [Paper](https://arxiv.org/pdf/2105.00243)——FedProto(Alignment of Local and Global Prototype Representations)

- **2023FGCS** "FedProc: Prototypical contrastive federated learning on non-IID data" [Paper](https://arxiv.org/pdf/2109.12273)——FedProc(Alignment of Local and Global Prototype Representations)
- **2020ICML** "SCAFFOLD:Stochastic Controlled Averaging for Federated Learning" [Paper](https://arxiv.org/pdf/1910.06378)——SCAFFOLD(Alignment of Local and Global Optimization Directions)
- **2021ICLR** "FEDERATED LEARNING BASED ON DYNAMIC REGULARIZATION" [Paper](https://arxiv.org/pdf/2111.04263)——FedDyn(Alignment of Local and Global Losses)

- **2022NeurIPS** "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning" [Paper](https://arxiv.org/pdf/2106.03097)——FedNTD(Alignment of Unseen Local Losses with Global Losses)

- **2021ICLR** "ADAPTIVE FEDERATED OPTIMIZATION" [Paper](https://arxiv.org/pdf/2003.00295)——FedOpt(Server-Side Aggregation Optimization)

- **2024CVPR** "Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity"[Paper](https://arxiv.org/pdf/2405.16585) & [Github](https://github.com/yuhangchen0/FedHEAL)——FedHEAL(Alignment of Local and Global Model Representations)

- **2023WACV** "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"[Paper](https://arxiv.org/pdf/2210.00912) & [Github](https://chenjunming.ml/proj/CCST)——CCST(Alignment of Local and Global Optimization Directions)

- **2023TMC** "FedFA: Federated Learning with Feature Anchors to Align Features and Classifiers for Heterogeneous Data" [Paper](https://arxiv.org/pdf/2211.09299)——FedFA(Alignment of Features and Classifiers)

- **2024AAAI** "CLIP-Guided Federated Learning on Heterogeneous and Long-Tailed Data" [Paper](https://arxiv.org/pdf/2312.08648)——CLIP As Backbond For FL

- **2023CVPR** "Rethinking Federated Learning with Domain Shift: A Prototype View" [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf) & [Github](https://github.com/WenkeHuang/RethinkFL/tree/main)——Cross-Domain Prototype Loss Alignment

- **2023ICLR** "FEDFA: FEDERATED FEATURE AUGMENTATION" [Paper](https://arxiv.org/pdf/2301.12995) & [Github](https://github.com/tfzhou/FedFA)——Class Prototype Gaussian Enhancement

- **2021ICLR** "FEDMIX: APPROXIMATION OF MIXUP UNDER MEAN AUGMENTED FEDERATED LEARNING" [Paper](https://arxiv.org/pdf/2107.00233)——Mixup For FL

- **2021PMLR** "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" [Paper](https://arxiv.org/pdf/2105.10056)——Data-Free Knowledge Distillation For FL

- **2017ICML** "Communication-Efficient Learning of Deep Networks from Decentralized Data" [Paper](https://arxiv.org/pdf/1602.05629)——FedAvg(Average aggregation)

- **2025ICLR** "Pursuing Better Decision Boundaries for Long-Tailed Object Detection via Category Information Amount" [Paper](https://arxiv.org/pdf/2502.03852)——IGAM Loss(Revise decision boundaries)

- **2025Entropy** "Trade-Offs Between Richness and Bias of Augmented Data in Long-Tailed Recognition" [Paper](https://www.mdpi.com/1099-4300/27/2/201)——EIG(Effectiveness of distributed gain)

---

## 💡 Citation

If you find our work useful, please cite it using the following BibTeX format:

<!-- ```bibtex
@inproceedings{ma2025geometric,
  title={Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning},
  author={Ma, Yanbiao and Dai, Wei and Huang, Wenke and Chen, Jiayi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
  note={Accepted}
}
``` -->

```bibtex
@article{ma2025geometric,
  title={Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning},
  author={Ma, Yanbiao and Dai, Wei and Huang, Wenke and Chen, Jiayi},
  journal={arXiv preprint arXiv:2503.06457},
  year={2025},
  url={https://arxiv.org/pdf/2503.06457}
}
```

## 📧 Contact

**For any questions or help, feel welcome to write an email to <br> 22012100039@stu.xidian.edu.cn or wdai@stu.xidian.edu.cn**
