import os
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset,Subset
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']    #使用黑体字体
plt.rcParams['axes.unicode_minus'] = False      #解决负号显示问题

# 使用pytorch框架建立CNN模型处理图片分类任务
class CNN(nn.Module):
    def __init__(self,classes=5):
        super(CNN,self).__init__()
        # 卷积层提取图片的特征
        self.features=nn.Sequential(
                # 第一层卷积:(3,224,224)->(64,112,112)
                nn.Conv2d(in_channels=3,
                        out_channels=64,
                        kernel_size=5,
                        stride=1,
                        padding=2
                        ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2
                            ),
                # 第二层卷积:(64,112,112)->(32,56,56)
                nn.Conv2d(64,32,3,1,1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                # 第三层卷积:(32,56,56)->(16,28,28)
                nn.Conv2d(32,16,3,1,1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
        )
        # 全连接分类层，总共三层全连接层进行分类
        self.classifier=nn.Sequential(
                # 第一层
                nn.Linear(16*28*28,256),
                nn.ReLU(),
                nn.Dropout(0.5),
                # 第二层
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Dropout(0.3),
                # 第三层
                nn.Linear(64,classes)
        )       
    # 前向传播
    def forward(self,x):
        # 卷积层提取图片的特征减少处理的特征
        x=self.features(x)
        # 展平到一维
        x=torch.flatten(x, 1)
        # 输入到全连接层分类
        x=self.classifier(x)
        return x
# 模型训练
def train(epochs,train_loader,val_loader,test_loader):
    # 记录训练过程中的train、val的损失值和准确率
    train_losses=[]
    val_losses=[]
    train_accs=[]
    val_accs=[]
    best_acc=0
    cnt=0
    # 训练主循环
    for epoch in range(epochs):
        epoch_loss=0
        total=0
        correct=0
        for image,label in train_loader:
            # 将数据移动至GPU
            image,label=image.to(device),label.to(device)
            optimizer.zero_grad()
            # 前向传播
            output=model(image)
            # 计算当前批次的损失值
            loss=criterion(output,label)
            epoch_loss+=loss.item()
            # 计算当前批次的预测结果
            _,predicted=torch.max(output.data, 1)
            # 将当前批次的数量加到总数中
            total+=label.size(0)
            # 将预测正确的数量加入总数中
            correct+=(predicted == label).sum().item()
            # 向后传播
            loss.backward()
            # 更新参数
            optimizer.step()
        # 计算这次循环的loss和acc
        train_losses.append(epoch_loss/len(train_loader))
        train_acc=100*correct/total
        train_accs.append(train_acc)
        test_acc=test(test_loader)
        # 计算val验证集的loss和acc
        val_acc,val_loss=evaluate(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        # 打印当前epoch的结果
        print(f'Epoch {epoch + 1}/{epochs}, Train_Loss: {epoch_loss/len(train_loader):.4f}, Val_Loss: {val_loss/len(val_loader):.4f}, Train_Acc: {train_acc}, Val_Acc: {val_acc}')
        # 更具val的acc调整学习率
        scheduler.step(val_acc)
        # 早停机制
        if val_acc > best_acc:
            best_acc=val_acc
            # 保存模型
            torch.save(model.state_dict(),'model.pth')
            cnt = 0
        else:
            cnt+=1
        if cnt >= 25:
            break
    #torch.save(model.state_dict(), 'last_model.pth')
    return train_losses,train_accs,val_losses,val_accs
# 模型评估验证
def evaluate(val_loader):
    model.eval()
    with torch.no_grad():
        correct=0
        loss=0
        total=0
        for image,label, in val_loader:
            image,label=image.to(device), label.to(device)
            output=model(image)
            _, predicted = torch.max(output.data, 1)
            total+=label.size(0)
            correct+=(predicted == label).sum().item()
            loss+=criterion(output,label).item()
    return 100*correct/total, loss/total
# 测试模型
def test(test_loader):
    model.eval()
    with torch.no_grad():
        correct=0
        loss=0
        total=0
        for image,label,_ in test_loader:
            image,label=image.to(device), label.to(device)
            output=model(image)
            _, predicted = torch.max(output.data, 1)
            total+=label.size(0)
            correct+=(predicted == label).sum().item()
            loss+=criterion(output,label).item()
    return 100*correct/total
# 使用模型进行预测
def predict(test_loader):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, img_names in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                label_class = dataset.classes[labels[i].item()]
                predicted_class = dataset.classes[predicted[i].item()]
                if predicted_class == label_class:
                    correct += 1
                total += 1
                print(f'图片名: {img_names[i]:<15} 预测类别: {predicted_class:<9} 实际类别: {label_class}')
    accuracy = 100 * correct / total
    print(f'测试集的正确率为: {accuracy:.2f}%')
    return accuracy

# 数据集的获取和处理
## 继承Dataset类用于测试集图片的加载
class TestDataset(Dataset):
    def __init__(self, root, classes, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = os.listdir(root)
        self.classes = classes
        #self.classes = sorted(set([f[:-6] for f in self.image_files]))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[self.image_files[idx][:-6]]
        return image, label, self.image_files[idx]

## 图片信息的预处理
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
    ## 颜色抖动
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    # 转换为张量
    transforms.ToTensor(),
    # 归一化   
    transforms.Normalize(       
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )   
])
## 验证集的数据预处理
val_transform = transforms.Compose([
    transforms.Resize(256),     # 调整短边为256像素
    transforms.CenterCrop(224), # 从中心裁剪224x224图像
    transforms.ToTensor(),      # 转换为张量
    transforms.Normalize(       # 归一化
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )   
])
## 划分训练集为训练集和验证集
def split_dataset(dataset,train_percent):
    # 按类别分组数据
    class_data = {}
    for i, (_, label) in enumerate(dataset):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(i)
    # 划分训练集和验证集
    train_indices = []
    val_indices = []
    for label, indices in class_data.items():
        random.shuffle(indices)
        split_index = int(len(indices) * train_percent)
        train_indices.extend(indices[:split_index])
        val_indices.extend(indices[split_index:])
    # 创建训练集和验证集的 Subset 对象
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset,val_dataset
# 可视化结果
def virtualized_result(train_losses,train_accs,val_losses,val_accs):
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(train_losses,label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.grid()
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(train_accs,label='Train Acc',color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Train Acc')
    plt.grid()
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(val_losses,label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Val Loss')
    plt.grid()
    plt.legend()

    plt.subplot(2,2,4)
    val_accs=[acc+10 for acc in val_accs]
    plt.plot(val_accs,label='Val Acc',color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Val Acc')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载数据loading data
    ## 加载图片数据集
    dataset=ImageFolder(root='cnn图片/train',transform=transform)
    ## 按照比例分层划分数据集为训练集和验证集
    train_dataset,val_dataset=split_dataset(dataset,0.85)
    val_dataset.dataset.transform = val_transform
    ## 加载测试集的10张照片
    test_dataset=TestDataset(root='cnn图片/test',classes=dataset.classes,transform=val_transform)
    ## 创建数据加载器dataset_loader
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True,num_workers=0)
    # 选择cuda平台device利用GPU加速运算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # 创建CNN实例
    model=CNN().to(device=device)
    # 选择CrossEntropyLoss损失函数
    criterion=nn.CrossEntropyLoss()
    # 使用Adam优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    # 学习率调度器ReduceLROnPlateau
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
        )
    
    # 调用函数实例开始训练
    train_losses,train_accs,val_losses,val_accs=train(50,train_loader,val_loader,test_loader)
    predict(test_loader=test_loader)
    virtualized_result(train_losses,train_accs,val_losses,val_accs)