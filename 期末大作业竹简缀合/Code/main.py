import os
import torch
from torch.utils.data import DataLoader

from SiameseNetworkMoudel import SiameseNN, IntegratedOpenCVSiameseNN, ContrastiveLoss, train
from DatasetLoader import BambooSlipsDataset, transform, SplitDataset
from IncompleteBambooSlipClassifier import IncompleteBambooSlipClassifier
from PIL import Image

# 选择cuda平台device利用GPU加速运算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 加载数据集
dataset = BambooSlipsDataset(root='..\yizhuihe',transform=transform)
train_dataset, val_dataset, test_dataset = SplitDataset(dataset)

# 创建数据加载器dataset_loader
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=0)
val_loader=DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=0)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True,num_workers=0)

# 创建SiameseNN模型或者IntegratedOpenCVSiameseNN模型
model = SiameseNN().to(device=device)
# model = IntegratedOpenCVSiameseNN().to(device=device)

# 对比损失函数
criterion = ContrastiveLoss()

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

if __name__ == '__main__':
    # 模型的训练
    threshold = train(model, device, optimizer, criterion, 20, train_loader, val_loader, test_loader)

    # 尝试在weizhuihe集中寻找可缀合的竹简
    analyzer = IncompleteBambooSlipClassifier()
    root = '../weizhuihe'
    incomplete_bambooships = []
    for img_name in os.listdir(root):
        img = Image.open(img_name).convert('RGB')
        is_complete = analyzer.analyze_slip(img)
        if is_complete:
            continue
        else :
            incomplete_bambooships.append(img_name)

    pairs = []
    for i in range(len(incomplete_bambooships)):
        for j in range(i, len(incomplete_bambooships)):
            img1 = Image.open(incomplete_bambooships[i])
            img2 = Image.open(incomplete_bambooships[j])
            img1, img2 = transform(img1), transform(img2)
            model.eval()
            img1, img2 = img1.to(device), img2.to(device)
            output1, output2 = model(img1, img2)
            dist = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            prediction = (dist < threshold).astype(int)
            if prediction:
                pairs.append((incomplete_bambooships[i], incomplete_bambooships[j]))
            else :
                continue
    print(pairs)
