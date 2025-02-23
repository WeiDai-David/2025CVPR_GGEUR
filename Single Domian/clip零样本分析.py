import torch
from PIL import Image
import open_clip
import torchvision.transforms as transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 CLIP 模型和预处理函数
backbone = 'ViT-B-32'  # 使用的 CLIP 模型骨干网络
pretrained_path = r'C:\Users\Games\Desktop\nature数据\open_clip_pytorch_model.bin'  # 预训练权重路径
model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained_path)
model.eval().to(device)  # 设置模型为评估模式并转移到设备

# 定义一个函数来进行图片的加载和推理
def predict_image(image_path, text_labels):
    # 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    
    # 将文本标签转化为 CLIP 模型的输入
    text_inputs = open_clip.tokenize(text_labels).to(device)
    
    # 获取图像和文本的特征向量
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
    
    # 进行特征向量的归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 计算图像和文本特征之间的相似度
    similarities = (image_features @ text_features.T).squeeze(0)
    
    # 找出与图像最匹配的文本标签
    best_match_idx = similarities.argmax().item()
    best_match_label = text_labels[best_match_idx]
    
    # 返回最匹配的文本标签和相似度
    return best_match_label, similarities[best_match_idx].item()

# 设置图片路径和待识别的文本标签
image_path = r'./data/office_caltech_10/dslr/bike/frame_0001.jpg'
text_labels = ["a bike", "a back_pack", "a calculator", "a headphones", "a keyboard", "a laptop_computer", "a monitor", "a mouse", "a mug", "a projector"]

# 执行推理并打印结果
best_match_label, similarity_score = predict_image(image_path, text_labels)
print(f"识别结果: {best_match_label}, 相似度: {similarity_score:.4f}")
