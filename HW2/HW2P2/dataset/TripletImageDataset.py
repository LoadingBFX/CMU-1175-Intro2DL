import os
import random
from PIL import Image
import torch
from torchvision import transforms


class TripletImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # 从 ImageFolder 获取类信息
        self.classes = image_folder.classes  # 从 ImageFolder 获取类别
        self.class_to_idx = image_folder.class_to_idx  # 从 ImageFolder 获取类别到索引的映射

        # 获取每个类别的样本路径
        self.class_samples = self._make_class_samples(image_folder.imgs)

    def _make_class_samples(self, imgs):
        class_samples = {cls: [] for cls in self.classes}
        for img_path, cls in imgs:
            class_samples[self.classes[cls]].append(img_path)  # 使用类别名称作为键
        return class_samples

    def __len__(self):
        return sum(len(samples) for samples in self.class_samples.values())

    def __getitem__(self, index):
        # 随机选择一个类别
        anchor_label = random.choice(self.classes)

        # 获取 anchor 图像
        anchor_img_path = random.choice(self.class_samples[anchor_label])

        # 获取正样本（同类别）
        positive_img_path = anchor_img_path
        while positive_img_path == anchor_img_path:  # 确保选择不同图像
            positive_img_path = random.choice(self.class_samples[anchor_label])

        # 获取负样本（不同类别）
        negative_label = anchor_label
        while negative_label == anchor_label:  # 确保选择不同类别
            negative_label = random.choice(self.classes)
        negative_img_path = random.choice(self.class_samples[negative_label])

        # 读取图像并转换为 RGB 格式
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')

        # 应用转换
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        # 返回 anchor、positive 和 negative 图像及其标签索引
        return anchor_img, positive_img, negative_img, self.class_to_idx[anchor_label]


# # 示例转换
# train_transforms = transforms.Compose([
#     transforms.Resize((128, 128)),  # 根据需要调整尺寸
#     transforms.ToTensor(),
# ])
#
# # 使用 ImageFolder 初始化数据集
# train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
#
# # 使用自定义 TripletImageDataset
# train_dataset = TripletImageDataset(train_dataset, transform=train_transforms)
#
# # 测试输出
# print("Number of classes:", len(train_dataset.classes))  # 输出类别数量
# print("Class to index mapping:", train_dataset.class_to_idx)  # 输出类别到索引的映射
# print("Shape of anchor image:", train_dataset[0][0].shape)  # 检查 anchor 图像形状
