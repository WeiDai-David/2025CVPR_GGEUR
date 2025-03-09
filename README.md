<!-- # 2025CVPR_GGEUR
联邦学习中几何引导的本地全局分布对齐

摘要：联邦学习中的数据异质性，其特征在于局部和全局分布之间的显著不一致，导致局部优化方向不同，并阻碍全局模型训练。现有的研究主要集中在优化本地更新或全局聚合，但这些间接方法在处理高度异构的数据分布时表现出不稳定性，特别是在标签偏斜和域偏斜共存的情况下。为了解决这个问题，我们提出了一种几何引导的数据增强方法，该方法侧重于在局部模拟全局嵌入分布。我们首先引入了嵌入分布几何形状的概念，并在隐私约束下解决了获取全局几何形状的挑战。随后，我们提出了GGEUR，它利用全局几何形状来引导新样本的生成，从而实现对理想全局分布更好的近似。在单域的情况下，我们通过基于全局几何形状的样本增强来提高模型泛化能力；在多域的情况下，我们进一步采用类原型来模拟跨域的全局分布。大量的实验结果表明，我们的方法在处理高度异构的数据时显著提升了性能，尤其是在标签倾斜，域倾斜，和二者共存的情况下。

关键词：联邦学习、数据异质、域泛化、感知流形 -->

# 2025CVPR_GGEUR
**项目整理中 待完成的工作 7/3/2025**：
(1):英文版本
(2):各个子工程脚本名的逻辑重构
(3):子工程内脚本的逻辑关系图 子工程间的逻辑关系图
(4):站在模型视野(流形空间)的角度将我们的思想以图的方式剖析
(5):更优美更逻辑的md表述

**Abstract:** Data heterogeneity in federated learning, characterized by a significant misalignment between local and global distributions, leads to divergent local optimization directions and hinders global model training. Existing studies mainly focus on optimizing local updates or global aggregation, but these indirect approaches demonstrate instability when handling highly heterogeneous data distributions, especially in scenarios where label skew and domain skew coexist. To address this, we propose a geometry-guided data generation method that centers on simulating the global embedding distribution locally. We first introduce the concept of the geometric shape of an embedding distribution and then address the challenge of obtaining global geometric shapes under privacy constraints. Subsequently, we propose GGEUR, which leverages global geometric shapes to guide the generation of new samples, enabling a closer approximation to the ideal global distribution. In singledomain scenarios, we augment samples based on global geometric shapes to enhance model generalization; in multidomain scenarios, we further employ class prototypes to simulate the global distribution across domains. Extensive experimental results demonstrate that our method significantly enhances the performance of existing approaches in handling highly heterogeneous data, including scenarios with label skew, domain skew, and their coexistence. Code published at: https://github.com/WeiDaiDavid/2025CVPR_GGEUR

**key word:** Federated Learning, Data Heterogeneity, Domain Generalization, Perceptual Manifold

## New Dataset Office-Home-LDS 

Dataset & constructor ： <a href="https://huggingface.co/datasets/WeiDai-David/Office-Home-LDS" target="_blank">Huggingface</a>

The dataset is organized as follows:
```text
Office-Home-LDS/
├── data/ 
│   └── Office-Home.zip        # Original raw dataset (compressed)
├── new_dataset/               # Processed datasets based on different settings
│   ├── Office-Home-0.1.zip    # Split with Dirichlet α = 0.1 (compressed)
│   ├── Office-Home-0.5.zip    # Split with Dirichlet α = 0.5 (compressed)
│   └── Office-Home-0.05.zip   # Split with Dirichlet α = 0.05 (compressed)
├── Dataset-Office-Home-LDS.py # Python script for processing and splitting Original raw dataset
└── README.md                  # Project documentation
```
## Engineering 

Environment：
```bash
conda create -n GGEUR python=3.9
conda activate GGEUR

```



## Single Domain

1.Dataset Partitioning
CIFAR (10 & 100) dataset:
Dataset index parsing:
```text
The CIFAR dataset is sorted by subsets of the same class, so the indexing process is directly arranged by the number of classes
```
Optional parameters:
```bash
num_clients integer Number of clients
alpha 浮点数 狄利克雷系数
min_require_size 整数 最小分配数
```
执行脚本:
```bash
python data_distribution_CIFAR-10.py
python data_distribution_CIFAR-100.py
python data_batch2index_images.py  检查划分
```
运行结果:
```bash
{数据集名称}/context/alpha={alpha}_class_counts.txt  统计划分后各类的总数(检查划分错误)
{数据集名称}/context/alpha={alpha}_class_indices.txt  根据数据集划分的数据索引
{数据集名称}/context/alpha={alpha}_client_indices.txt  各客户端分配到的数据索引
{数据集名称}/context/alpha={alpha}_client_class_distribution.txt  各客户端下各类的数据分布
{数据集名称}/images/alpha={alpha}_label_distribution_heatmap.png  各客户端分配各类数量的热力图
```
TinyImageNet数据集:
数据集索引解析：
```bash
TinyImageNet数据集来源于ImageNet的200个类,同时并非线性排列,下面我们将进行数据集的重构
TinyImageNet数据集的结构:
│
├── train/
│   └── n01443537/   # 类别文件夹
│       └── images/  # 类别对应的训练图像
│           └── image_001.JPEG
│           └── image_002.JPEG
│           └── ...
├── val/
│   ├── images/      # 验证图像
│   └── val_annotations.txt  # 图像标签映射文件
│
├── test/
│    └── images/      # 测试图像
│
├── wnids.txt 类别对应的id
│
└── words.txt 类别对应的标签

测试集没有标签,因此我们将验证集作为测试集,训练集遵循ImageNet的文件夹层次结构,验证集的图片
都存储在val/images中,val/val_annotations.txt存储了验证集图片和标签id的映射关系
通过Reorganized_TinyImageNet_Val.py脚本将验证集的结构和训练集结构对齐 

│
├── train/
│   └── n01443537/   # 类别文件夹
│       └── images/  # 类别对应的训练图像
│           └── image_001.JPEG
│           └── image_002.JPEG
│           └── ...
├── new_val/
│   └── n01443537/   # 类别文件夹
│       └── images/  # 类别对应的验证图像
│           └── image_001.JPEG
│           └── image_002.JPEG
│           └── ...

接下来,我们需要对标签id(n+8位数)进行索引,为了统一,我们将标签id去除n后进行排序,以确定索引类顺序
这部分工作在data_distribution_TinyImageNet.py(划分训练集)TinyImageNet_Val.py(划分验证集)中体现
至此,我们完成TinyImageNet数据集的重构,并未接下来的索引划分工作铺垫
```
可选参数:
```bash
num_clients 整数 客户端数量
alpha 浮点数 狄利克雷系数
min_require_size 整数 最小分配数
```
执行脚本:
```bash
python reorganized_TinyImageNet_val.py  重构验证集
python TinyImageNet_val_index.py  验证集索引转化
python data_distribution_TinyImageNet.py
python TinyImageNet_val_index_tag_img_matching_test.py 检查验证集处理
python TinyImageNet_val_features.py 提取验证集特征
```
运行结果:
```bash
{数据集名称}/context/alpha={alpha}_class_counts.txt  统计划分后各类的总数(检查划分错误)
{数据集名称}/context/alpha={alpha}_class_indices.txt  根据数据集划分的数据索引
{数据集名称}/context/alpha={alpha}_client_indices.txt  各客户端分配到的数据索引
{数据集名称}/context/alpha={alpha}_client_class_distribution.txt  各客户端下各类的数据分布
{数据集名称}/images/alpha={alpha}_label_distribution_heatmap.png  各客户端分配各类数量的热力图
{数据集名称}/context/class_map.txt  训练集类索引和标签id的映射
{数据集名称}/val_context/class_map.txt  验证集类索引和标签id的映射
{数据集名称}/val_context/val_indices.npy  验证集数据索引
{数据集名称}/val_context/val_labels.npy  验证集数据索引对应的标签
{数据集名称}/class_{class_label}_val_indices.npy  验证集各类的数据索引
```

2.交叉索引
我们已经得到了CIFAR-10、CIFAR-100、TinyImageNet三个数据集,各个客户端划分的数据索引,类的数据索引
通过将两者进行交叉索引,我们能得到各个客户端下各个类的数据索引,以CIFAR-10为例：10个客户端*10个类=100个交叉索引文件

可选参数:
```bash
dataset 字符串 数据集名称(三个数据集)
alpha 浮点数 狄利克雷系数
```
执行脚本:
```bash
python client_class_cross_index.py
```
运行结果:
```bash
{数据集名称}/client_class_indices/alpha={alpha}_{dataset}_client_{client_id}_class_{class_id}_indices.npy
```

3.特征提取
我们已经得到了三个数据集下,各个客户端下,各个类的索引文件,使用CLIP作为Backbond,我们对索引文件逐个进行特征提取
得到对应的特征文件和标签文件

可选参数:
```bash
dataset 字符串 数据集名称(三个数据集)
alpha 浮点数 狄利克雷系数
```
执行脚本:
```bash
python client_class_clip_features2tensor.py
```
运行结果:
```bash
{数据集名称}/features/initial/alpha={alpha}_class_{class_idx}_client_{client_idx}/final_embeddings.npy
{数据集名称}/features/initial/alpha={alpha}_class_{class_idx}_client_{client_idx}/labels.npy
```

4.全局分布
4.1 全局分布的客户端成分
在1.数据集划分中得到的alpha={alpha}_client_class_distribution.txt,里面存储各客户端下各类的数据分布,
通过类数量对客户端进行排序,根据阈值比例确定能近似全局分布的客户端集合

可选参数:
```bash
threshold 浮点数 阈值比例
dataset 字符串 数据集名称(三个数据集)
alpha 浮点数 狄利克雷系数
```
执行脚本:
```bash
python client-guided_set.py
```
运行结果:
```bash
{数据集名称}/context/alpha={alpha}_selected_clients_for_each_class.txt
```
4.2 客户端集合的特征(可省略)
在3.特征提取中得到各个客户端下各个类的特征文件,在4.1全局分布的客户端成分中得到各个类全局分布的客户端组成
将后者对应的特征文件从前者中抽取
可选参数:
```bash
dataset 字符串 数据集名称(三个数据集)
alpha 浮点数 狄利克雷系数
```
执行脚本:
```bash
python client-guided_clip_tensor.py
```
4.3 全局分布的表示
如果客户端集合只有一个客户端，则直接对特征矩阵进行协方差矩阵的计算，如果客户端集合由多个客户端组成，则分别对
特征矩阵进行协方差矩阵的计算，最后将多个协方差矩阵进行聚合，得到聚合协方差矩阵
可选参数:
```bash
dataset 字符串 数据集名称(三个数据集)
alpha 浮点数 狄利克雷系数
```
执行脚本:
```bash
python clip_tensor2aggregate_covariance_matrix.py
```
运行结果:
```bash
{数据集名称}/features/alpha={alpha}_cov_matrix/class_{idx}_cov_matrix.npy
```

5.几何引导的数据增强
现在我们已经得到了全局分布的骨架(协方差矩阵),将协方差矩阵分解得到特征值和特征向量(几何方向),利用几何方向
引导客户端中的原始样本进行数据增强
可选参数:
```bash
dataset 字符串 数据集名称(三个数据集)
alpha 浮点数 狄利克雷系数
```
执行脚本:
```bash
python cov_matrix_generate_features.py
```
运行结果:
```bash
{数据集名称}/features/alpha={alpha}_complete/final_embeddings_filled.npy
{数据集名称}/features/alpha={alpha}_complete/labels_filled.npy
```

6.单客户端训练
在非联邦学习架构下,本地训练原始样本和增强样本,比较性能差异

可选参数:
```bash
alpha 浮点数 狄利克雷系数
client_idx 整数 客户端编号
batch_size 整数 批次大小
learning_rate 浮点数 学习率
num_epochs 整数 训练次数
```
执行脚本:
```bash
python MLP_10.py  CIFAR-10数据集训练
python MLP_100.py  CIFAR-100数据集训练
python MLP_200.py  TinyImageNet数据集训练
```

7.联邦架构FedAvg训练
在最简单的联邦学习架构FedAvg(简单平均聚合)下,训练原始样本模型和增强样本模型,,比较性能差异
可选参数:
```bash
alpha 浮点数 狄利克雷系数
communication_rounds 整数 通信轮数
local_epochs 整数 本地训练次数
client_count 整数 客户端数量
batch_size 整数 批次大小
learning_rate 浮点数 学习率
```
执行脚本:
```bash
python FL_MLP_10.py  CIFAR-10数据集训练
python FL_MLP_100.py  CIFAR-100数据集训练
python FL_MLP_200.py  TinyImageNet数据集训练
```

## 跨域情况 
我们分别对Digits、PACS、office_caltech_10数据集进行实验,而后提出新数据集Office-Home-LDS
Digits:
1.数据集介绍：
Digits 数据集包含四个不同的域（domains）的数据，类别为数字0-9这十种类别
(1): MNIST（Mixed National Institute of Standards and Technology）：不同人手写的数字（0-9）
(2): SVHN（Street View House Numbers）：谷歌街景的门牌号裁剪成数字（0-9）
(3): USPS（United States Postal Service）：美国邮政编号裁剪的数字（0-9）
(4): SynthDigits（Synthetic Digits）：合成方法生成的彩色（RGB）的数字（0-9）
2.数据集划分
我们遵循联邦跨域以往的工作设置：一个客户端划分一个域，同时各客户端按照比例进行随机划分
执行脚本:
```bash
python data_distribution_digits.py
```
运行结果:
```bash
output_indices/{域名称}/client_{域对应的客户端编号}_indices.npy  该域内客户端划分的索引
output_indices/{域名称}/combined_class_indices.npy  该域内各类的索引集合
output_indices/client_combined_class_distribution.txt  各域分配给对应客户端各类的数据分布
output_indices/dataset_report.txt  各域分配给对应客户端的总数(检查划分错误)
```
3.交叉索引
我们已经得到某域对应的客户端在该域划分的数据索引,以及该域的类别索引,通过交叉索引,我们可以得到该域对应客户端的各类的索引文件
执行脚本:
```bash
python 交叉索引.py
```
运行结果:
```bash
output_client_class_indices/{域名称}/client_{域对应的客户端编号}_class_{0~9}_indices.npy
```
4.训练集特征提取
我们已经得到了四个域对应的四个客户端下,,各个类的索引文件,使用CLIP作为Backbond,我们对索引文件逐个进行特征提取得到对应的特征文件和标签文件
可选参数:
```bash
datasets 字符串 域名称(四个可选域)
report_file 字符串 域和客户端的映射文件
```
执行脚本:
```bash
python 训练集特征.py
```
运行结果:
```bash
clip_features/{域名称}/client_{域对应的客户端编号}_class_{0~9}_original_features.npy
clip_features/{域名称}/client_{域对应的客户端编号}_class_{0~9}_labels.npy
```
5.测试集特征提取
可选参数:
```bash
datasets 字符串 域名称(四个可选域)
```
执行脚本:
```bash
python 测试集特征.py
```
运行结果:
```bash
clip_test_features/{域名称}/{域名称}_test_features.npy
clip_test_features/{域名称}/{域名称}_test_labels.npy
```
6.提取原型
通过前面得到的客户端类别索引文件，可以提取对应的类原型
执行脚本:
```bash
python 提取原型.py
```
运行结果:
```bash
clip_prototypes/{域名称}/client_{域对应的客户端编号}_class_{0~9}_prototype.npy
```
7.几何方向
在流形空间的视角，跨域的本质是类对应的流形分布发生横向的偏移，几何方向并没有变化，因此我们可以利用多域的合并特征来表示几何形状
```bash
report_file 字符串 域和客户端的映射文件
```
执行脚本:
```bash
python 聚合协方差矩阵4x10=10.py
```
运行结果:
```bash
cov_matrix_output/class_{0~9}_cov_matrix.npy
```
8.几何引导的数据增强
现在我们已经得到了分布的几何方向和多个域对应的类原型，在客户端进行数据增强，客户端本域，进行单域数据增强策略，非本域，则以类原型为中心进行几何方向的增强，这样即使在客户端视角，也能学习到跨域的特征，从而有效的缓解域偏移带来的偏差
执行脚本:
```bash
python 扩充-放大聚合协方差矩阵-类原型-类中心.py
```
运行结果:
```bash
argumented_clip_features/{域名称}/client_{域对应的客户端编号}_class_{0~9}/final_embeddings_filled.npy
argumented_clip_features/{域名称}/client_{域对应的客户端编号}_class_{0~9}/labels_filled.npy
```
9.不同联邦架构训练
执行脚本:
```bash
python FedAvg联邦原始特征.py 
python FedAvg联邦补全特征.py 
python FedNTD联邦原始特征.py
python FedNTD联邦补全特征.py
python FedOpt联邦原始特征.py
python FedOpt联邦补全特征.py
python FedProx联邦原始特征.py
python FedProx联邦补全特征.py
python MOON联邦原始特征.py
python MOON联邦补全特征.py
python FedDyn联邦原始特征.py
python FedDyn联邦补全特征.py
python FedProto联邦原始特征.py
python FedProto联邦补全特征.py
python SCAFFOLD联邦原始特征.py
python SCAFFOLD联邦补全特征.py
```

PACS和office_caltech_10是小样本数据集和少分类任务
少分类任务-分类难度低
类间差异大-有利于区分
小样本数据集中划分训练集测试集-训练集和测试集相似-隐式提高精度
同域类划分相对平衡(因为是按比例随机划分，这是跨域划分的通病)

随着自监督学习的发展，面向以clip、dino为代表的Backbond时，PACS和office_caltech_10这类跨域数据集的挑战被大大的削弱，因此我们迫切的需要一种更符合现实场景且更具有挑战的数据集，因此我们提出了Office-Home-LDS，它在Office-Home数据集(65分类)的基础上融合了跨域和数据异质，显然，这更符合真实世界。

PACS:
1.数据集介绍：
PACS 数据集包含四个不同的域（domains）的数据，包含七个类别:Dog、Elephant、Giraffe、Guitar、Horse、House、Person
(1): P（Photo）: 真实照片
(2): A（Art Painting）: 艺术绘画
(3): C（Cartoon）: 卡通
(4): S（Sketch）: 素描
2.数据集划分
我们遵循联邦跨域以往的工作设置：一个客户端划分一个域，同时各客户端按照比例从训练集中随机划分,对于没有划分训练集和测试集的数据集，遵循以往工作的比例进行划分
执行脚本:
```bash
python data_distribution_digits.py
```
运行结果:
```bash
./output_indices/{域名称}/train_train_indices.npy 该域数据集划分的训练集索引
./output_indices/{域名称}/test_test_indices.npy 该域数据集划分的测试集索引
./output_indices/{域名称}/client_{域对应的客户端编号}_indices.npy  该域内客户端划分的索引
./output_indices/{域名称}/class_indices.npy  该域内各类的索引集合
./output_indices/client_combined_class_distribution.txt  各域分配给对应客户端各类的数据分布
```
3.交叉索引
我们已经得到某域对应的客户端在该域划分的数据索引,以及该域的类别索引,通过交叉索引,我们可以得到该域对应客户端的各类的索引文件
执行脚本:
```bash
python 交叉索引.py
```
运行结果:
```bash
output_client_class_indices/{域名称}/client_{域对应的客户端编号}_class_{0~6}_indices.npy
```
4.训练集特征提取
我们已经得到了四个域对应的四个客户端下,,各个类的索引文件,使用CLIP作为Backbond,我们对索引文件逐个进行特征提取得到对应的特征文件和标签文件
执行脚本:
```bash
python 训练集特征.py
```
运行结果:
```bash
clip_pacs_train_features/{域名称}/client_{域对应的客户端编号}_class_{0~6}_original_features.npy
clip_pacs_train_features/{域名称}/client_{域对应的客户端编号}_class_{0~6}_labels.npy
```
5.测试集特征提取
执行脚本:
```bash
python 测试集特征.py
```
运行结果:
```bash
clip_test_features/{域名称}/{域名称}_test_features.npy
clip_test_features/{域名称}/{域名称}_test_labels.npy
```
6.不同联邦架构训练
执行脚本:
```bash
python FedAvg联邦原始特征.py 
python FedNTD联邦原始特征.py
python FedOpt联邦原始特征.py
python FedProx联邦原始特征.py
python MOON联邦原始特征.py
python FedDyn联邦原始特征.py
python FedProto联邦原始特征.py
python SCAFFOLD联邦原始特征.py
```

office_caltech_10:
1.数据集介绍：
office_caltech_10 数据集包含四个不同的域（domains）的数据，包含十个类别:Backpack、Calculator、
Headphones、Keyboard、Laptop、Monitor、Mouse、Mug、Projector、Bike
(1): Amazon (A): 来自 Amazon 的商品图片
(2): Webcam (W): 通过网络摄像头拍摄的图像
(3): DSLR (D): 由数码单反相机拍摄的图像
(4): Caltech-256 (C): 从 Caltech-256 数据集中选取的图像
2.数据集划分
我们遵循联邦跨域以往的工作设置：一个客户端划分一个域，同时各客户端按照比例从训练集中随机划分,对于没有划分训练集和测试集的数据集，遵循以往工作的比例进行划分
执行脚本:
```bash
python data_distribution_office_caltech_10.py
```
运行结果:
```bash
output_indices/{域名称}/train_train_indices.npy 该域数据集划分的训练集索引
output_indices/{域名称}/test_test_indices.npy 域数据集划分的测试集索引
output_indices/{域名称}/client_{域对应的客户端编号}_indices.npy  该域内客户端划分的索引
output_indices/{域名称}/class_indices.npy  该域内各类的索引集合
output_indices/client_combined_class_distribution.txt  各域分配给对应客户端各类的数据分布
```
3.交叉索引
我们已经得到某域对应的客户端在该域划分的数据索引,以及该域的类别索引,通过交叉索引,我们可以得到该域对应客户端的各类的索引文件
执行脚本:
```bash
python 交叉索引.py
```
运行结果:
```bash
output_client_class_indices/{域名称}/client_{域对应的客户端编号}_class_{0~9}_indices.npy
```
4.训练集特征提取
我们已经得到了四个域对应的四个客户端下,,各个类的索引文件,使用CLIP作为Backbond,我们对索引文件逐个进行特征提取得到对应的特征文件和标签文件
执行脚本:
```bash
python 训练集特征.py
```
运行结果:
```bash
clip_pacs_train_features/{域名称}/client_{域对应的客户端编号}_class_{0~9}_original_features.npy
clip_pacs_train_features/{域名称}/client_{域对应的客户端编号}_class_{0~9}_labels.npy
```
5.测试集特征提取
执行脚本:
```bash
python 测试集特征.py
```
运行结果:
```bash
clip_test_features/{域名称}/{域名称}_test_features.npy
clip_test_features/{域名称}/{域名称}_test_labels.npy
```
6.不同联邦架构训练
执行脚本:
```bash
python FedAvg联邦原始特征.py 
python FedNTD联邦原始特征.py
python FedOpt联邦原始特征.py
python FedProx联邦原始特征.py
python MOON联邦原始特征.py
python FedDyn联邦原始特征.py
python FedProto联邦原始特征.py
python SCAFFOLD联邦原始特征.py
```

Office-Home-LDS:
1.数据集介绍：
Office-Home-LDS 数据集包含四个不同的域（domains）的数据，有65个类别
(1): Art： 手绘、素描或油画风格的图像
(2): Clipart： 来自网络的卡通和剪贴画图像
(3): Product：	产品图片或商品展示图
(4): Real World： 现实场景中拍摄的图像
2.数据集划分
我们遵循联邦跨域以往的工作设置：一个客户端划分一个域，同时各客户端按照狄利克雷矩阵中的比例从训练集中随机划分,对于没有划分训练集和测试集的数据集，遵循以往工作的比例进行划分
执行脚本:
```bash
python data_distribution_Office_Home_LDS.py
```
运行结果:
```bash
./output_indices/{域名称}/train_train_indices.npy 该域数据集划分的训练集索引
./output_indices/{域名称}/test_test_indices.npy 该域数据集划分的测试集索引
./output_indices/{域名称}/client_{域对应的客户端编号}_indices.npy 该域内客户端划分的索引
./output_indices/{域名称}/class_indices.npy 该域内各类的索引集合
./output_indices/client_combined_class_distribution.txt 各域分配给对应客户端各类的数据分布
```
3.交叉索引
我们已经得到某域对应的客户端在该域划分的数据索引,以及该域的类别索引,通过交叉索引,我们可以得到该域对应客户端的各类的索引文件
执行脚本:
```bash
python 交叉索引.py
```
运行结果:
```bash
output_client_class_indices/{域名称}/client_{域对应的客户端编号}_class_{0~64}_indices.npy
```
4.训练集特征提取
我们已经得到了四个域对应的四个客户端下,,各个类的索引文件,使用CLIP作为Backbond,我们对索引文件逐个进行特征提取得到对应的特征文件和标签文件
执行脚本:
```bash
python 训练集特征.py
```
运行结果:
```bash
clip_office_home_train_features/{域名称}/client_{域对应的客户端编号}_class_{0~64}_original_features.npy
clip_office_home_train_features/{域名称}/client_{域对应的客户端编号}_class_{0~64}_labels.npy
```
5.测试集特征提取
执行脚本:
```bash
python 测试集特征.py
```
运行结果:
```bash
clip_test_features/{域名称}/{域名称}_test_features.npy
clip_test_features/{域名称}/{域名称}_test_labels.npy
```
6.提取原型
通过前面得到的客户端类别索引文件，可以提取对应的类原型
执行脚本:
```bash
python 提取原型.py
```
运行结果:
```bash
./office_home_prototypes/{域名称}/client_{域对应的客户端编号}_class_{0~64}_prototype.npy
```
7.几何方向
在流形空间的视角，跨域的本质是类对应的流形分布发生横向的偏移，几何方向并没有变化，因此我们可以利用多域的合并特征来表示几何形状
```bash
report_file 字符串 域和客户端的映射文件
```
执行脚本:
```bash
python 聚合协方差矩阵4x65=65.py
```
运行结果:
```bash
cov_matrix_output/class_{0~64}_cov_matrix.npy
```
8.几何引导的数据增强
现在我们已经得到了分布的几何方向和多个域对应的类原型，在客户端进行数据增强，客户端本域，进行单域数据增强策略，非本域，则以类原型为中心进行几何方向的增强，这样即使在客户端视角，也能学习到跨域的特征，从而有效的缓解域偏移带来的偏差
执行脚本:
```bash
python 扩充-放大聚合协方差矩阵-类原型-类中心-封顶.py.py
```
运行结果:
```bash
argumented_clip_features/{域名称}/client_{域对应的客户端编号}_class_{0~64}/final_embeddings_filled.npy
argumented_clip_features/{域名称}/client_{域对应的客户端编号}_class_{0~64}/labels_filled.npy
```
9.不同联邦架构训练
执行脚本:
```bash
python FedAvg联邦原始特征.py 
python FedAvg联邦补全特征.py 
python FedNTD联邦原始特征.py
python FedNTD联邦补全特征.py
python FedOpt联邦原始特征.py
python FedOpt联邦补全特征.py
python FedProx联邦原始特征.py
python FedProx联邦补全特征.py
python MOON联邦原始特征.py
python MOON联邦补全特征.py
python FedDyn联邦原始特征.py
python FedDyn联邦补全特征.py
python FedProto联邦原始特征.py
python FedProto联邦补全特征.py
python SCAFFOLD联邦原始特征.py
python SCAFFOLD联邦补全特征.py
```
## Learning Trajectory (Updating...)

When I completed this project, I was a third-year undergraduate student. 🌿 I will share my learning trajectory and how to efficiently and comprehensively develop expertise in a specific field. 🌊 I believe that the most effective approach is to start by identifying high-quality review articles from top-tier journals. 📚 After forming a comprehensive understanding of the field, I recommend selecting detailed papers from the references cited in these outstanding reviews, focusing on those that align with the direction of our current work for in-depth study. 🔍 This process resembles a leaf with its veins hollowed out — our process of understanding is akin to a flood flowing through the leaf, with the central vein serving as the core from which knowledge selectively branches out in all directions. 🚀

+ **2023Tpami**  "Deep Long-Tailed Learning: A Survey"[Paper](https://arxiv.org/pdf/2304.00685)——Review on Long-Tailed Learning 

+ **2024Tpami** "Vision-Language Models for Vision Tasks: A Survey" [Paper](https://arxiv.org/pdf/2304.00685) & [Github](https://github.com/jingyi0000/VLM_survey)——Review on Vision-Language Large Models

+ **2024Tpami**  "Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark" [Paper](https://arxiv.org/pdf/2311.06750) & [Github](https://github.com/WenkeHuang/MarsFL)——Review on Federated Learning

+ **2021CVPR**  "Model-Contrastive Federated Learning" [Paper](https://arxiv.org/pdf/2103.16257) & [Github](https://github.com/QinbinLi/MOON)——MOON(Alignment of Local and Global Model Representations)

+ **2022AAAI** "FedProto: Federated Prototype Learning across Heterogeneous Clients"[Paper](https://arxiv.org/pdf/2105.00243)——FedProto(Alignment of Local and Global Prototype Representations)

+ **2023FGCS** "FedProc: Prototypical contrastive federated learning on non-IID data" [Paper](https://arxiv.org/pdf/2109.12273)——FedProc(Alignment of Local and Global Prototype Representations)
  
+ **2020ICML** "SCAFFOLD:Stochastic Controlled Averaging for Federated Learning"[Paper](https://arxiv.org/pdf/1910.06378)——SCAFFOLD(Alignment of Local and Global Optimization Directions)
  
+ **2021ICLR** "FEDERATED LEARNING BASED ON DYNAMIC REGULARIZATION"[Paper](https://arxiv.org/pdf/2111.04263)——FedDyn(Alignment of Local and Global Losses)

+ **2022NeurIPS** "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning" [Paper](https://arxiv.org/pdf/2106.03097 )——FedNTD(Alignment of Unseen Local Losses with Global Losses)

+  **2021ICLR** "ADAPTIVE FEDERATED OPTIMIZATION"[Paper](https://arxiv.org/pdf/2003.00295)——FedOpt(Server-Side Aggregation Optimization)

+ **2024CVPR**  "Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity"[Paper](https://arxiv.org/pdf/2405.16585) & [Github](https://github.com/yuhangchen0/FedHEAL)——FedHEAL(Alignment of Local and Global Model Representations)

+ **2023WACV**  "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"[Paper](https://arxiv.org/pdf/2210.00912) & [Github](https://chenjunming.ml/proj/CCST)——CCST(Alignment of Local and Global Optimization Directions)

+ **2023TMC**  "FedFA: Federated Learning with Feature Anchors to Align Features and Classifiers for Heterogeneous Data"[Paper](https://arxiv.org/pdf/2211.09299)——FedFA(Alignment of Features and Classifiers)

+ **2024AAAI** "CLIP-Guided Federated Learning on Heterogeneous and Long-Tailed Data"[Paper](https://arxiv.org/pdf/2312.08648)——CLIP As Backbond For FL

+ **2023CVPR** "Rethinking Federated Learning with Domain Shift: A Prototype View"[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf) & [Github](https://github.com/WenkeHuang/RethinkFL/tree/main)——Cross-Domain Prototype Loss Alignment

+ **2023ICLR** "FEDFA: FEDERATED FEATURE AUGMENTATION" [Paper](https://arxiv.org/pdf/2301.12995) & [Github](https://github.com/tfzhou/FedFA)——Class Prototype Gaussian Enhancement

+ **2021ICLR** "FEDMIX: APPROXIMATION OF MIXUP UNDER MEAN AUGMENTED FEDERATED LEARNING" [Paper](https://arxiv.org/pdf/2107.00233)——Mixup For FL

+ **2021PMLR** "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"  [Paper](https://arxiv.org/pdf/2105.10056)——Data-Free Knowledge Distillation For FL

+ **2017ICML** "Communication-Efficient Learning of Deep Networks from Decentralized Data" [Paper](https://arxiv.org/pdf/1602.05629)——FedAvg(Average aggregation)





