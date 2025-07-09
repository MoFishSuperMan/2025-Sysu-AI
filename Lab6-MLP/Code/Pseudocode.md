```python
Algorithm: Multi-Layer-Perceptron
Input:  X,y  // 特征矩阵以及目标值向量
        η   // 学习率
        epochs  // 训练轮数
        g(x)    // 激活函数
Return: model   // 训练好的模型
Initialize: W1
            b1
            W2
            b2
for epoch=0 to epochs do
    // 前向传播
    Z1 ← W1·X + b1
    A1 ← g(Z1)
    Z2 ← W2·A1 + b2
    Loss ← Loss(Z2)
    // 反向传播
    dZ2 ← (Z2-y)/n
    dW2 ← np.dot(dZ2,self.A1.T)
    db2 ← np.sum(dZ2,axis=1,keepdims=True)
    dA1 ← np.dot(self.W2.T,dZ2)
    dZ1 ← dA1 * self.A1 * (1-self.A1)
    dW1 ← np.dot(dZ1,X.T)
    db1 ← np.sum(dZ1,axis=1,keepdims=True)
end for
```
