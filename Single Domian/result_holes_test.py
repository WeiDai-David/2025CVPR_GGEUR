import os
from PIL import Image
import torch
import numpy as np
import open_clip
from tqdm import tqdm

def process_text_file(file_path, output):
    number_to_word = {}
    # Read the file and store the mapping from number to word(utf-8)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if output == 'classes':
                if line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        number = int(parts[0])
                        description = parts[1].split(',')
                        first_word = description[0]
                        number_to_word[number] = first_word
            elif output == 'classes_ID':
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        number = parts[0]
                        description = parts[1].split(' ')
                        first_word = description[0]
                        number_to_word[first_word] = number
    return number_to_word

class_label_path = 'F:\ImageNet_test\class_label.txt'
W_ID = process_text_file(class_label_path, 'classes_ID')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpeg', '.jpg')):  # 支持JPEG格式
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # 确保图片为RGB格式
                    images.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images

def extract_features(model, preprocess, images, device="cuda", batch_size=16):
    all_features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        inputs = torch.stack([preprocess(image) for image in batch_images]).to(device)
        with torch.no_grad():
            features = model.encode_image(inputs)
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def process_dataset(root_dir, output_dir, model_name='ViT-B-32', model_path=None, root='huggingface'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if root == None:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, model_path=model_path, device=device, jit=False)  # 加载模型
    elif root == 'huggingface':
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name)  # 加载模型

    model.to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    flag = 0

    for folder_name in tqdm(os.listdir(root_dir), desc="Processing folders"):
        folder_path = os.path.join(root_dir, folder_name)
        output_dir_01 = os.path.join(output_dir, str(W_ID.get(folder_name)))
        if not os.path.exists(output_dir_01):
            os.makedirs(output_dir_01)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            if folder_name == 'n03457902':
                flag = 1
            if flag == 1:
                images = load_images_from_folder(folder_path)
                if images:
                    features = extract_features(model, preprocess, images, device=device)
                    model_name = model_name.replace('hf-hub:', '').replace('/', '-')
                    np.save(os.path.join(output_dir_01, f"{model_name}_features.npy"), features)
                    print(f"{features.shape}")
                else:
                    print(f"No images found in {folder_name}")

# 使用示例
root_dir = 'F:\IN2012img_lbw'  # 修改为您的数据集根目录
output_dir = r'F:\results'  # 修改为您想要保存特征文件的目录
model_name='hf-hub:apple/DFN5B-CLIP-ViT-H-14-378'
model_path=r"E:\pythonProject1\open_clip-main\b32_fullcc2.5b.pt"
process_dataset(root_dir, output_dir,model_name=model_name, root='huggingface')
#'hf-hub:timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k'
