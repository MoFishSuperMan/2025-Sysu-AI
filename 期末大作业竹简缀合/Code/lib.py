import matplotlib.pyplot as plt
from torchvision.models import resnet18

plt.rcParams['font.sans-serif'] = ['SimHei']    #使用黑体字体
plt.rcParams['axes.unicode_minus'] = False      #解决负号显示问题

# 可视化结果
def VirtualiseResult(train_losses, val_losses, test_acc):
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(train_losses,label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(val_losses,label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Val Loss')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.plot(test_acc,label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Test Acc')
    plt.grid()
    plt.legend()
    plt.show()