import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#练习1还包括一个房屋价格数据集，其中有2个变量（房子的大小，卧室的数量）和目标（房子的价格）。 我们使用我们已经应用的技术来分析数据集。
#========================== Part 1: Feature Normalization=========================
print('加载数据......')
path = './data/ex1data2.txt'
data = pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])
print(data.head())

#特征归一化
data = (data - data.mean()) / data.std()
print(data.head())

##在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度
data.insert(0,'One',1)

#设置变量X(训练数据)和y（目标变量）
cols = data.shape[1]# 获取列数 - 3
X = data.iloc[:,0:cols-1] #所有行，不包含最后一列
y = data.iloc[:,cols-1:cols] #所有行,只包含最后一列
# print(X.head())#只显示前5个数据
# print(y.head())#只显示前5个数据

#np.matrix()函数用于从类数组对象或数据字符串返回矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))#初始化参数,1x3矩阵

#计算代价损失
def computeCost(X,y,theta):
    num = len(X)
    inner = np.power(((X * theta.T)-y),2)
    cost = np.sum(inner) / (2*num)
    return cost


#批量梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X,y,theta)

    return theta,cost

alpha = 0.01#学习速率α
iters = 1000#要执行的迭代次数。

#执行线性回归在数据集上
g,cost = gradientDescent(X,y,theta,alpha,iters)

#获得模型的代价
computeCost(X,y,g)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
