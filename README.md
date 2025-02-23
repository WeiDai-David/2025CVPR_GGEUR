# 2025CVPR_GGEUR
项目开源，整体思路解析

## 工作过程问题总结（逐步整理更新）
（1）：基础视觉语言大模型对Office-Caltech跨域数据集的强大影响

动机：实验过程中，通过分配客户端组给Office-Caltech实现数据异质，对比2024cvpr&2023cvpr，3214客户端组划分策略，每个客户端抽取该域数据集0.2的数据量，引入clip作为backbond（对比resnet10）fedavg聚合效果98%，采取更极端划分，只分配0.01数据量，发现结果还是很高，修改客户端组为1111，数据量0.01，各个数据域划分数据为1，训练效果仍然很高

推测原因：

  1:训练集和测试集图像之间极度相似，即使在少样本情况下，仍可以使训练效果极好（clip特征相似度高） 
  
  2:类间差异性大，即使clip下文本图像相似度不太高，图片与图片之间相似度不高，但是十个类之间很好区分。
  
后面可以补充一个小实验，对Office-Caltech的四个数据集分别进行clip测试，具体实验：拿一个数据集举例，首先是十个类各取一张图片，作为训练图片参照，然后拿一张训练图片参照与十个类各随机抽的一张图片，一共十张图片做相似度计算，可以证明即使样本只有一个，分类效果也很好

（2）：clip backbond下分类任务数量的探究

动机：在工作中我们发现在clip作为backbond的情况下，对多个跨域数据集（digits、Office-Caltech、pacs）进行特征提取，参照2024cvpr&2023cvpr&2024Tpami的设置，联邦（Fedavg）架构下精度高达97%，我们进行了更极端的训练划分，多round收敛后仍然能达到95%，最极端情况下，我们设置各个域只被一个客户端划分，各个客户端的各个类只划分一个样本，联邦聚合后精度仍有90%
基于此现象，我们希望对当前多个主流联邦架构，进行clip+联邦架构的性能分析，分析传统设置，少样本设置，零样本设置分析

推测原因：

 1:诸如clip、dino v2这样的视觉语言基础大模型，在面临多分类任务场景，表现出60-70较低的精度，在对于十分类cifar-10、少分类跨域数据集而言，表现出较高的精度，得益于clip图像解码器的相似度分析，由于训练clip的策略是对比学习，开放学习，持续学习，导致clip对于万物都有一定的识别，直观的就是对于Office-Caltech的随便一张bike图片，使用文本图片相似度、图片余弦相似度分析，都能取得不低的效果，这种效果在少分类问题，尤其是类间差异大的情况下将会被进一步放大
 
 2:跨域数据集中虽然提出了跨域，但是不同域的相同类仍然具有较大的相似性和一致性，特别的就是Office-Caltech的样本

 据此我们希望在实验后，提出少分类跨域问题将被诸如clip这样的大模型解决，我们将进一步研究多分类跨域问题



## 学习路线（持续整理中...）
+ 2024Tpami "Vision-Language Models for Vision Tasks: A Survey" [Github URL](https://github.com/jingyi0000/VLM_survey)——视觉语言大模型综述
<!-- Author：Jingyi Zhang, Jiaxing Huang, Sheng Jin, Shijian Lu, School of Computer Science and Engineering, Nanyang Technological University, Singapore. -->

+ 2023Tpami "Deep Long-Tailed Learning: A Survey"——长尾学习综述
<!-- Author：Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, Fellow, IEEE, and Jiashi Feng -->

+ 2024Tpami  "Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark" [Github URL](https://github.com/WenkeHuang/MarsFL)——联邦学习综述
<!-- Author：Wenke Huang, Mang Ye, Senior Member, IEEE , Zekun Shi, Guancheng Wan, He Li, Bo Du, SeniorMember, IEEE, Qiang Yang, Fellow, IEEE -->

+ 2021CVPR  "Model-Contrastive Federated Learning" [Github URL](https://github.com/QinbinLi/MOON)——原型对比学习联邦架构MOON
<!-- Author：Qinbin Li, Bingsheng He, National University of Singapore, Dawn Song, UC Berkeley -->

+ 2024CVPR  "Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity" [Github URL](https://github.com/yuhangchen0/FedHEAL)——跨域联邦学习
<!-- Author：Yuhang Chen, Wenke Huang, Mang Ye, National Engineering Research Center for Multimedia Software, School of Computer Science, Wuhan Universit -->

+ 2023CVPR  "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer" [Github URL](https://chenjunming.ml/proj/CCST)——域泛化
<!-- Author:Junming Chen，Meirui Jiang, Qi Dou, Qifeng Chen HKUST CUHK -->

+ 2023IEEE TMC  "FedFA: Federated Learning with Feature Anchors to Align Features and Classifiers for Heterogeneous Data"——特征对齐
<!-- Author：Tailin Zhou, Graduate Student Member, IEEE, Jun Zhang, Fellow, IEEE, and Danny H.K. Tsang, Life Fellow, IEEE -->

+ 2024IJCV "Geometric Prior Guided Feature Representation Learning for Long-Tailed Classificatio"——几何先验引导
<!-- Yanbiao Ma, Licheng Jiao，Fang Liu, Shuyuan Yang, Xu Liu, Puhua Chen, School of Artificial Intelligence, Xidian University, Xi’an -->

+ 2024AAAI "CLIP-Guided Federated Learning on Heterogeneous and Long-Tailed Data"——CLIP-backbond
<!-- Jiangming Shi, Shanshan Zheng, Xiangbo Yin, Yang Lu, Yuan Xie, Yanyun Qu1, Institute of Artificial Intelligence, Xiamen University -->

+ 2023FGCS "FedProc: Prototypical Contrastive Federated Learning on Non-IID data"——单域场景类原型损失对比
单域场景中利用特征信息在服务器端聚合全局类原型, 然后下发客户端与局部类原型对比损失, 约束局部模型更新方向, 相对的约束局部模型有偏

+ 2023CVPR "Rethinking Federated Learning with Domain Shift: A Prototype View" [Github URL](https://github.com/WenkeHuang/RethinkFL/tree/main)——跨域场景类原型损失对比
跨域场景中利用不同域的特征信息聚合全局类原型，然后下发客户端与局部单域类原型进行域对齐损失，让本地模型偏向学习到跨域的信息，相对的缓解域偏移的偏差

+ 2023ICLR "FEDFA: FEDERATED FEATURE AUGMENTATION" [Github URL](https://github.com/tfzhou/FedFA)——类原型高斯分布生成
本地客户端生成类原型，假定数据遵循高斯分布，并且类原型位于中央，基于类原型进行样本扩充


## 笔记存储

1：在 MOON 中，虽然使用了对比学习损失和近端正则项来约束本地模型训练，但是在参数聚合阶段，MOON 依然使用了与 FedAvg 类似的聚合策略来进行全局模型的更新。因此，MOON 的联邦学习架构在参数聚合部分与 FedAvg 是类似的，即仍然使用“全局模型等于所有客户端模型的平均值”来更新全局模型的权重。
MOON 的主要区别在于它在客户端模型的本地训练过程中增加了对比学习损失和近端项约束，而不是在聚合策略上与 FedAvg 完全不同。MOON 仍然使用 FedAvg 样式的权重聚合，因为这是标准的联邦学习参数聚合方法。

2：在 FedProx 中，虽然客户端更新时加入了近端项，但全局模型的参数聚合方式仍然可以使用 FedAvg 聚合策略。这是因为 FedProx 本质上是在客户端的本地训练过程中加入了对全局模型的约束，即引入了近端项，以减少本地模型和全局模型的偏差。然而，在全局模型更新时，仍然可以使用 FedAvg 进行参数聚合。

3：FedProto 是一种联邦学习的变种，通过在联邦学习过程中引入原型学习，使每个客户端可以使用全局共享的类原型来进行训练。它与其他联邦方法不同，联邦聚合是通过计算客户端本地数据的类原型并与全局原型对比，来调整模型的训练。
每个客户端将计算并更新本地类原型。全局模型聚合所有客户端的类原型。训练过程中通过类原型来进行对比学习。

分类损失 (classification_loss)：模型输出和真实标签的交叉熵损失。
原型对比损失 (proto_loss)：本地模型的权重与类原型之间的差异，用于进行原型对比学习。
近端项 (proximal_term)：本地模型与全局模型的参数差异，使用 λ 控制其权重，类似 FedProx 中的近端正则化。

4：FedOpt 是基于 FedAvg 的一种改进方法，它引入了服务器端的优化器，而不是单纯的在服务器端进行模型权重的平均。FedOpt 的目的是使用优化器在服务器端更新全局模型，增强对异质数据的适应性。
通过服务器端优化器的引入，FedOpt 可以加快模型的收敛速度，尤其是在异质数据集上。
允许在服务器端进行更灵活的全局优化，而不仅仅依赖客户端本地的更新。

5：FedNTD 是一种基于知识蒸馏的联邦学习方法，旨在应对不同客户端之间数据分布差异较大的问题。FedNTD 通过知识蒸馏的方式，使每个客户端能够从其他客户端学习，从而提升全局模型的泛化能力。



## 工程解析
当前为线性运行逻辑，项目重构的工作还在进行，未来我们将推出重构版本

环境
```bash
整理中...
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
数据集名称/context/alpha={alpha}_class_counts.txt  统计划分后各类的总数(检查划分错误)
数据集名称/context/alpha={alpha}_class_indices.txt  根据数据集划分的数据索引
数据集名称/context/alpha={alpha}_client_indices.txt  各客户端分配到的数据索引
数据集名称/context/alpha={alpha}_client_class_distribution.txt  各客户端下各类的数据分布
数据集名称/images/alpha={alpha}_label_distribution_heatmap.png  各客户端分配各类数量的热力图
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
数据集名称/context/alpha={alpha}_class_counts.txt  统计划分后各类的总数(检查划分错误)
数据集名称/context/alpha={alpha}_class_indices.txt  根据数据集划分的数据索引
数据集名称/context/alpha={alpha}_client_indices.txt  各客户端分配到的数据索引
数据集名称/context/alpha={alpha}_client_class_distribution.txt  各客户端下各类的数据分布
数据集名称/images/alpha={alpha}_label_distribution_heatmap.png  各客户端分配各类数量的热力图
数据集名称/context/class_map.txt  训练集类索引和标签id的映射
数据集名称/val_context/class_map.txt  验证集类索引和标签id的映射
数据集名称/val_context/val_indices.npy  验证集数据索引
数据集名称/val_context/val_labels.npy  验证集数据索引对应的标签
数据集名称/class_{class_label}_val_indices.npy  验证集各类的数据索引
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
数据集名称/client_class_indices/alpha={alpha}_{dataset}_client_{client_id}_class_{class_id}_indices.npy
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
数据集名称/features/initial/alpha={alpha}_class_{class_idx}_client_{client_idx}/final_embeddings.npy
数据集名称/features/initial/alpha={alpha}_class_{class_idx}_client_{client_idx}/labels.npy
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
数据集名称/context/alpha={alpha}_selected_clients_for_each_class.txt
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
数据集名称/features/alpha={alpha}_cov_matrix/class_{idx}_cov_matrix.npy
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
数据集名称/features/alpha={alpha}_complete/final_embeddings_filled.npy
数据集名称/features/alpha={alpha}_complete/labels_filled.npy
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
PACS:
office_caltech_10:
Office-Home-LDS:

