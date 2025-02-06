import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from scipy.spatial.distance import cosine
import numpy as np
import glob

def load_model():
    # 1. 加载 ResNet-50 预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()  # 设为评估模式
    return model

model = load_model()
# 2. 移除 ResNet 的最后一层 (FC)，只保留 `avgpool` 层
resnet50_feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# 3. 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. 读取并处理图像
def resnet_extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # 增加 batch 维度
    with torch.no_grad():
        features = resnet50_feature_extractor(img)
    return features.squeeze().numpy()  # 输出 (2048,)

# 5. 计算特征向量的余弦相似度

image_list = ["E:\\tmp\\scene_change_frames_001.png", "E:\\tmp\\scene_change_frames_002.png",
              "E:\\tmp\\scene_change_frames_003.png", "E:\\tmp\\scene_change_frames_004.png",]

feature1 = resnet_extract_features(image_list[1])
feature2 = resnet_extract_features(image_list[3])

similarity = 1 - cosine(feature1, feature2)
print(f"余弦相似度: {similarity}")


# 加载 VGG16 预训练模型
vgg = models.vgg16(pretrained=True)
vgg.eval()

# 只取倒数第二层的 `fc2` 层输出 (4096 维)
vgg_feature_extractor = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])

# 预处理 + 提取特征
def extract_vgg_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = vgg_feature_extractor(vgg.features(img).view(img.shape[0], -1))
    return features.squeeze().numpy()

feature1 = extract_vgg_features(image_list[1])
feature2 = extract_vgg_features(image_list[3])

similarity = 1 - cosine(feature1, feature2)
print(f"余弦相似度: {similarity}")



# **批量提取多个图像的 VGG 特征**
def extract_vgg_features_batch(image_paths):
    images = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)

    # **构造 batch**
    batch = torch.stack(images)  # (batch_size, 3, 224, 224)
    
    with torch.no_grad():
        # 先提取卷积特征，再展平
        features = vgg.features(batch).view(batch.shape[0], -1)
        features = vgg_feature_extractor(features)  # (batch_size, 4096)
    
    return features.numpy()

# **批量处理多个图片**
image_paths = glob.glob("E:\\tmp\\*.png")  # 例如，处理 frames 文件夹中的所有 .jpg 图片
features = extract_vgg_features_batch(image_paths)  # 形状: (num_images, 4096)

# **计算所有图片之间的相似度**
num_images = len(features)
similarities = np.zeros((num_images, num_images))

for i in range(num_images):
    for j in range(i + 1, num_images):  # 只计算上三角矩阵，减少重复计算
        similarities[i, j] = 1 - cosine(features[i], features[j])
        similarities[j, i] = similarities[i, j]  # 对称矩阵

# **输出相似度矩阵**
print("图像相似度矩阵：")
print(similarities)
