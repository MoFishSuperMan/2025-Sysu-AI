import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *

plt.rcParams['font.sans-serif'] = ['SimHei']    #使用黑体字体
plt.rcParams['axes.unicode_minus'] = False      #解决负号显示问题

class MLP:
    def __init__(self,data,input_dim,hidden_dim,output_dim,learn_rate):
        # 加载数据
        self.X,self.y=self.load_data(data)
        # MLP实例的参数初始化
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.learn_rate=learn_rate
        # 记录训练过程的损失值
        self.predictions=[]
        self.test_losses=[]
        # 偏置值、权重矩阵的初始化
        np.random.seed(50)
        self.W1=np.random.randn(hidden_dim,input_dim)* np.sqrt(2 / input_dim)
        self.b1=np.zeros((hidden_dim,1))* np.sqrt(2 / input_dim)
        self.W2=np.random.randn(output_dim,hidden_dim)* np.sqrt(2 / hidden_dim)
        self.b2=np.zeros((output_dim,1))* np.sqrt(2 / hidden_dim)

    def load_data(self,filename):
        '''加载数据集'''
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        data_size=data.shape[0]
        # 特征矩阵
        X=data[ : , :4].T
        # 标签向量
        y=data[ : , 4].reshape(1,-1)
        # 划分数据集 训练集：数据集 = 8：2
        index=int(data_size*0.8)
        train_X=X[ : , :index]
        train_y=y[ : , :index]
        self.test_X,self.test_X_MIN,self.test_X_MAX=self.MIN_MAX(X[ : , index: ])
        self.test_y,self.test_y_MIN,self.test_y_MAX=self.MIN_MAX(y[ : , index: ]) 
        X,self.X_MIN,self.X_MAX=self.MIN_MAX(train_X)
        y,self.y_MIN,self.y_MAX=self.MIN_MAX(train_y)
        return X,y

    def MIN_MAX(self,data):
        '''MIN_MAX归一化'''
        data_min = np.min(data, axis=1, keepdims=True)
        data_max = np.max(data, axis=1, keepdims=True)
        return (data - data_min) / (data_max - data_min + 1e-8),data_min,data_max
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def MSE(self,y,y_hat):
        return np.mean((y-y_hat)**2)/2
    def dMSE(self,y,y_hat):
        return (y_hat-y)/y.size
    
    def ReLU(self,x):
        return np.maximum(0,x)

    def forward(self,X,y):
        '''前向传播'''
        # 隐藏层
        self.Z1=np.dot(self.W1,X)+self.b1
        self.A1=self.ReLU(self.Z1)
        #self.A1=self.sigmoid(self.Z1)
        # 输出层，回归问题采用线性输出，即不进行激活函数激活
        self.Z2=np.dot(self.W2,self.A1)+self.b2
        return self.Z2

    def backward(self,X,y):
        '''反向传播'''
        # 输出层grad
        dZ2=self.dMSE(y,self.Z2)
        dW2=np.dot(dZ2,self.A1.T)
        db2=np.sum(dZ2,axis=1,keepdims=True)
        # 隐藏层grad
        dA1=np.dot(self.W2.T,dZ2)
        dZ1=dA1 * (self.Z1 > 0)
        #dZ1=dA1 * self.A1 * (1-self.A1)
        dW1=np.dot(dZ1,X.T)
        db1=np.sum(dZ1,axis=1,keepdims=True)
        # update参数
        self.W1-=self.learn_rate*dW1
        self.W2-=self.learn_rate*dW2
        self.b1-=self.learn_rate*db1
        self.b2-=self.learn_rate*db2

    def fit(self,epochs=1000,tolerant_error=1e-6):
        '''基于数据集训练模型'''
        self.losses=[]
        self.test_losses=[]
        best_loss=float('inf')
        cnt=0
        for epoch in range(epochs):
            if epoch == 0:
                self.predictions.append(self.predict(self.test_X,self.test_y))
            # 学习率递减机制
            self.learn_rate*=0.999
            # 前向传播
            cur_output=self.forward(self.X,self.y)
            # 反向传播
            self.backward(self.X,self.y)
            # 计算损失函数
            loss=self.MSE(self.y,cur_output)
            self.losses.append(loss)
            self.predict(self.test_X,self.test_y)
            # 早停机制
            if self.test_losses[-1] < best_loss:
                if np.abs(self.test_losses[-1]-best_loss) < tolerant_error:
                    cnt+=1
                    if cnt >= 200:
                        break
                best_loss=self.test_losses[-1]
            else :
                cnt+=1
                if cnt >= 200:
                    break
            # 每隔100次训练输出一次损失值
            if epoch % 1 ==0:
                print(f'Epoch: {epoch}: Loss: {loss}')
        self.predictions.append(self.predict(self.test_X,self.test_y))

    def predict(self,X,y):
        '''测试集测试'''
        # 隐藏层
        Z1=np.dot(self.W1,X)+self.b1
        A1=self.ReLU(Z1)
        #A1=self.sigmoid(Z1)
        # 输出层，回归问题采用线性输出，即不进行激活函数激活
        Z2=np.dot(self.W2,A1)+self.b2
        loss=self.MSE(self.test_y,Z2)
        self.test_losses.append(loss)
        return Z2
    
    def Loss_curve(self):
        '''损失值Loss变化曲线'''
        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.plot(self.losses[10:])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("训练集的Loss变化曲线")
        #plt.ylim(0,0.7)
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(self.test_losses[10:],color='brown')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("测试集的Loss变化曲线")
        #plt.ylim(0,0.7)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def draw_result(self):
        plt.figure(figsize=(10,6))
        for i,y_pred in enumerate(self.predictions):
            plt.subplot(1,2,i+1)
            x = np.linspace(0, 5e5, 100)
            y1=x
            y2=x+1e5
            y3=x-1e5
            if i!=0 :
                plt.plot(x, y1, label='y = x', color='red')
                #plt.plot(x, y2, color='red',linestyle='--')
                #plt.plot(x, y3, color='red',linestyle='--')
                plt.xlim(-0.5e5,5.5e5)
                plt.ylim(-0.5e5,5.5e5)
                plt.legend()
            plt.scatter(y_pred*(self.test_y_MAX-self.test_y_MIN)+self.test_y_MIN,self.test_y*(self.test_y_MAX-self.test_y_MIN)+self.test_y_MIN,s=5)
            #plt.scatter(y_pred*(self.y_MAX-self.y_MIN)+self.y_MIN,self.y*(self.y_MAX-self.y_MIN)+self.y_MIN,s=5)
            plt.xlabel("y_pred")
            plt.ylabel("y_true")

            if i == 0:
                plt.title(f'训练初期模型预测测试集的真实值和预测值散点图')
            else:
                plt.title(f'训练完成后模型预测测试集的真实值和预测值散点图')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model=MLP(data="MLP_data.csv",input_dim=4,hidden_dim=16,output_dim=1,learn_rate=0.1)
    model.fit(epochs=10000)
    model.Loss_curve()
    model.draw_result()
    #print(model.test_losses[-1])
