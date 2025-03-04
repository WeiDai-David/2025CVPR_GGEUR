# 2025CVPR_GGEUR
联邦学习中几何引导的本地全局分布对齐

摘要：联邦学习中的数据异质性，其特征在于局部和全局分布之间的显著不一致，导致局部优化方向不同，并阻碍全局模型训练。现有的研究主要集中在优化本地更新或全局聚合，但这些间接方法在处理高度异构的数据分布时表现出不稳定性，特别是在标签偏斜和域偏斜共存的情况下。为了解决这个问题，我们提出了一种几何引导的数据增强方法，该方法侧重于在局部模拟全局嵌入分布。我们首先引入了嵌入分布几何形状的概念，并在隐私约束下解决了获取全局几何形状的挑战。随后，我们提出了GGEUR，它利用全局几何形状来引导新样本的生成，从而实现对理想全局分布更好的近似。在单域的情况下，我们通过基于全局几何形状的样本增强来提高模型泛化能力；在多域的情况下，我们进一步采用类原型来模拟跨域的全局分布。大量的实验结果表明，我们的方法在处理高度异构的数据时显著提升了性能，尤其是在标签倾斜，域倾斜，和二者共存的情况下。

关键词：联邦学习、数据异质、域泛化、感知流形

## 工程解析
当前为线性运行逻辑，项目重构的工作还在进行，未来我们将推出重构版本

环境 
```bash
conda create -n GGEUR python=3.9
conda activate GGEUR

```
## 单域情况 

1.数据集划分
CIFAR(10,100)数据集：
数据集索引解析：
```bash
CIFAR数据集按照同类子集依次排列,因此构造索引过程直接按照类数量进行索引排列
```
可选参数:
```bash
num_clients 整数 客户端数量
alpha 浮点数 狄利克雷系数
min_require_size 整数 最小分配数
```
执行脚本:
```bash
python data_distribution_CIFAR-10.py
python data_distribution_CIFAR-100.py
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
python Reorganized_TinyImageNet_Val.py  重构验证集
python TinyImageNet_Val.py  验证集索引转化
python data_distribution_TinyImageNet.py
ps python TinyImageNet_Val_Index_tag_image_matching_test.py 检查验证集处理
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
python client_class_index.py
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
python best_client_guidance_100.py
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
python guide_clip_tensor.py
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
python clip_image_tensor2aggregate_covariance_matrix_100.py
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
python cov_matrix_generate_features.py_new_100.py
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
python MLP.py  CIFAR-10数据集训练
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
python new-FL-MLP-10-images-all.py  CIFAR-10数据集训练
python new-FL-MLP-100-images-all.py  CIFAR-100数据集训练
python new-FL-MLP-200-images-all.py  TinyImageNet数据集训练
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



PACS:
office_caltech_10:
Office-Home-LDS:

