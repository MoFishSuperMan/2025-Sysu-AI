import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
import random
import cv2
from tqdm import tqdm
import numpy as np
from lib import VirtualiseResult

random.seed(100)
torch.cuda.manual_seed_all(100)
torch.manual_seed(100)

class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN,self).__init__()
        # 采用resnet18预训练模型作为孪生神经网络的主干网络
        model = resnet18(pretrained=True)
        # 卷积层
        self.features=nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

        # 全连接层
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )
    
    def forward_once(self,x):
        # 卷积层提取图片的特征减少处理的特征
        x = self.features(x)
        # 
        x = self.classifier(x)
        return x

    def forward(self,x1,x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

# 对比损失函数（Contrastive Loss）
class ContrastiveLoss(nn.Module):
    '''ContrastuveLossFunction'''
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        label: 标签 (batch_size, 1)，1表示可拼缀，0表示不可拼缀
        """
        # 欧式距离
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1, keepdim=True))
        # 正样本:可拼缀的损失：距离的平方
        piecable_loss = label * torch.pow(euclidean_distance, 2)
        # 负样本:不可拼缀的损失：max(margin - distance, 0)的平方
        inpiecable_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # 平均损失
        loss = torch.mean(piecable_loss + inpiecable_loss)
        return loss

# OpenCV特征提取层
class CVFeatureExtractionLayer(nn.Module):
    def __init__(self, feature_type='sift', max_features=100, input_size=(3, 224, 224)):
        super(CVFeatureExtractionLayer, self).__init__()
        self.feature_type = feature_type
        self.max_features = max_features
        self.input_size = input_size
        
        # 初始化OpenCV特征提取器
        if feature_type == 'sift':
            self.extractor = cv2.SIFT_create(nfeatures=max_features)
        elif feature_type == 'surf':
            self.extractor = cv2.SURF_create()
        elif feature_type == 'orb':
            self.extractor = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
        
        # 计算特征维度
        if feature_type == 'orb':
            self.feature_dim = 32 * max_features
        else:
            self.feature_dim = 128 * max_features
    
    def forward(self, x):
        """
        前向传播方法
        x: 输入图像张量 (batch_size, channels, height, width)
        """
        batch_size = x.size(0)
        device = x.device
        # 初始化输出张量
        cv_features = torch.zeros(batch_size, self.feature_dim, device=device)
        # 对批次中的每个图像进行特征提取
        for i in range(batch_size):
            # 将张量转换为PIL图像再转回OpenCV格式
            img_tensor = x[i].cpu()  # 移到CPU进行处理
            img_pil = transforms.ToPILImage()(img_tensor)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # 转换为灰度图
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # 检测关键点和计算描述符
            kp, des = self.extractor.detectAndCompute(gray, None)
            # 处理特征描述符
            if des is None:
                # 如果没有检测到特征，返回全零向量
                feature_vector = torch.zeros(self.feature_dim, device=device)
            else:
                # 确保特征数量不超过max_features
                if len(des) > self.max_features:
                    des = des[:self.max_features]
                else:
                    # 不足则填充零
                    padding = np.zeros((self.max_features - len(des), des.shape[1]), dtype=des.dtype)
                    des = np.vstack((des, padding))
                # 展平为一维向量
                feature_vector = torch.from_numpy(des.flatten()).float().to(device)
            # 保存到输出张量
            cv_features[i] = feature_vector
        return cv_features

    def get_feature_dim(self):
        """获取特征维度"""
        return self.feature_dim

# 集成OpenCV特征提取的混合孪生神经网络
class IntegratedOpenCVSiameseNN(nn.Module):
    def __init__(self, feature_type='sift', max_features=100, attention_type='self'):
        super(IntegratedOpenCVSiameseNN, self).__init__()
        self.attention_type = attention_type
        # OpenCV特征提取层
        self.cv_feature_extractor = CVFeatureExtractionLayer(
            feature_type=feature_type,
            max_features=max_features
        )
        cv_feature_dim = self.cv_feature_extractor.get_feature_dim()
        # 图像分支 - 使用ResNet18
        model = resnet18(pretrained=True)
        self.image_features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )
        self.image_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # OpenCV特征分支
        self.cv_features = nn.Sequential(
            nn.Linear(cv_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # 自注意力机制
        if attention_type == 'self':
            self.self_attention = nn.Sequential(
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
                nn.Softmax(dim=1)
            )
        # 特征融合分支
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward_once(self, x):
        """处理单个图像"""
        # 提取OpenCV特征
        cv_features = self.cv_feature_extractor(x)
        # 处理图像输入
        img_features = self.image_features(x)
        img_repr = self.image_classifier(img_features)
        # 处理CV特征输入
        cv_repr = self.cv_features(cv_features)
        # 使用注意力机制融合特征
        if self.attention_type == 'self':
            # 计算自注意力权重
            combined_features = torch.stack([img_repr, cv_repr], dim=1)  # [batch_size, 2, 256]
            attention_weights = self.self_attention(combined_features)  # [batch_size, 2, 1]
            # 应用注意力权重
            weighted_features = torch.sum(combined_features * attention_weights, dim=1)  # [batch_size, 256]
        else:
            # 默认使用简单平均
            weighted_features = (img_repr + cv_repr) / 2.0
        # 最终特征表示
        final_repr = self.fusion(weighted_features)
        return final_repr
    
    def forward(self, x1, x2):
        """处理图像对"""
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

# 训练模型
def train(model, device, optimizer, criterion, epochs, train_loader, val_loader, test_loader):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    Acc = []
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for (img_pair, label) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            img1, img2 = img_pair
            label = label.float().view(-1, 1).to(device)
            img1, img2 = img1.to(device), img2.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            output1, output2 = model(img1, img2)
            # 计算当前批次的损失值
            loss = criterion(output1, output2, label)
            train_loss += loss.item()
            # 向后传播
            loss.backward()
            # 更新参数
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (img_pair, label) in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                img1, img2 = img_pair
                label = label.float().view(-1, 1).to(device)
                img1, img2 = img1.to(device), img2.to(device)
                
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_siamese_model.pth")
        best_threshold, best_accuracy = FindOptimalThreshold(model, device, val_loader)
        best_accuracy = test(model, device, best_threshold, test_loader)
        Acc.append(best_accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    VirtualiseResult(train_losses[1:], val_losses[1:], Acc[1:])
    return best_threshold

def FindOptimalThreshold(model, device, val_loader):
    model.eval()
    distances = []
    labels = []

    with torch.no_grad():
        for (img_pair, label) in val_loader:
            img1, img2 = img_pair
            label = label.float().view(-1, 1).to(device)
            img1, img2 = img1.to(device), img2.to(device)
            # 前向传播
            output1, output2 = model(img1, img2)
            # 欧式距离
            dist = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            distances.extend(dist.cpu().numpy())
            labels.extend(label.cpu().numpy())
    distances = np.array(distances)
    labels = np.array(labels)

    thresholds = np.linspace(0, 4, 200)
    best_threshold = 0
    best_accuracy = 0

    for threshold in thresholds:
        predictions = (distances < threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def test(model, device, threshold, test_loader):
    '''
    选取两个例子来测试模型的效果，一个正例子，一个反例子
    '''
    model.eval()
    distances = []
    labels = []
    with torch.no_grad():
        for (img_pair, label) in test_loader:
            img1, img2 = img_pair
            label = label.float().view(-1, 1).to(device)
            img1, img2 = img1.to(device), img2.to(device)
            
            output1, output2 = model(img1, img2)
            dist = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            distances.extend(dist.cpu().numpy())
            labels.extend(label.cpu().numpy())
    distances = np.array(distances)
    labels = np.array(labels)
    predictions = (distances < threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    return accuracy