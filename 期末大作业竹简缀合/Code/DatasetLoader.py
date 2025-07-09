import os
from PIL import Image
from torch.utils.data import Dataset,Subset
from torchvision import transforms
import random
from itertools import combinations


# 图片集的读取与处理
class BambooSlipsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.groups = self.load_group()
        self.postiive_pairs = self.generate_positive_pairs()
        self.negative_pairs = self.generate_negative_pairs()
        self.image_pairs = self.postiive_pairs + self.negative_pairs
        self.labels = [1] * len(self.postiive_pairs) + [0] * len(self.negative_pairs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1_name, img2_name = self.image_pairs[idx]
        label = self.labels[idx]
        img1 = Image.open(img1_name).convert('RGB')
        img2 = Image.open(img2_name).convert('RGB')
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return (img1, img2), label

    def load_group(self):
        '''loading group'''
        groups = []
        for group_name in os.listdir(self.root_dir):
            group = os.path.join(self.root_dir, group_name)
            if os.path.isdir(group):
                image_files = [os.path.join(group,f) for f in os.listdir(group) if f.endswith(('.png', '.jpg', 'jpeg'))]
                if image_files:
                    groups.append(image_files)
        return groups
    
    def generate_positive_pairs(self):
        '''正样本对：已缀合的图片'''
        positive_pairs = []
        for group_images in self.groups:
            if len(group_images) < 2:
                continue

            pairs = list(combinations(group_images, 2))
            positive_pairs.extend(pairs)
        return positive_pairs[0:int(len(positive_pairs) * 0.5)]

    def generate_negative_pairs(self, num_negative_pairs=None):
        negative_pairs = []
        # 如果未指定数量，默认与正样本数量相同
        if num_negative_pairs is None:
            num_negative_pairs = int(len(self.postiive_pairs) * 0.5)
        for _ in range(num_negative_pairs):
            # 随机选择两个不同的组
            group1, group2 = random.sample(self.groups, 2)
            # 从每个组中随机选择一张图片
            img1 = random.choice(group1)
            img2 = random.choice(group2)
            negative_pairs.append((img1, img2))
        return negative_pairs

# 图片信息的预处理
transform = transforms.Compose([
    # 调整短边为256像素
    transforms.Resize(256),
    # 数据增强
    ## 随机裁剪并缩放
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ## 50%概率水平翻转
    transforms.RandomHorizontalFlip(),
    ## 随机旋转±15度
    transforms.RandomRotation(15), 
    # 转换为张量
    transforms.ToTensor(),
    # 归一化   
    transforms.Normalize(       
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )   
])

# 划分数据集 train:val:test 8:1:1
def SplitDataset(dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    # random.shuffle(indices)
    # 创建Subset对象
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:train_size + val_size])
    test_dataset = Subset(dataset, indices[train_size + val_size:])
    return train_dataset, val_dataset, test_dataset