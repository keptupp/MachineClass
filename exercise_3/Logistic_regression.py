"""
   在本部分练习中，你将建立一个逻辑回归模型，以预测学生是否被大学录取
假设你是大学部门的管理员，并且您想根据每位申请人的两次考试成绩来确定他们的录取机会。
您具有以前申请人的历史数据，可以用作逻辑回归的训练集。对于每个训练示例，您都有两次
考试的申请人分数和入学决定。
   您的任务是建立一个分类模型，根据这两次考试的分数来估算申请人的录取概率。
"""
import pandas as pd
import numpy as np
import seaborn as sns #seaborn 是在matplotlib的基础上进行了更高级的API封装
import matplotlib.pyplot as plt

print('加载数据........')
path = './data/ex2data1.txt'
data = pd.read_csv(path,names = ['exam1','exam2','admitted'])
# print(data.head())#查看前5行的数据
# print(data.describe())#查看数据的属性

#绘制散点图
sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
           height=6,
           fit_reg=False,
           scatter_kws={"s": 50}
          )

sns.lmplot('exam1','exam2',hue='admitted',data=data,height=6,fit_reg=False,scatter_kws={"s":50})
# plt.show()

#读取特征值
def get_X(df):
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """

    """
    DataFrame是python 中 Pandas库中的一种数据结构，它类似excel,是一种二维表。
      --下面的代码是以字典的方式创建
    """
    ones = pd.DataFrame({'ones':np.ones(len(df))})#ones是m行1列的dateframe，以字典的方式创建
    # print(ones)#第ones标志的列的元素全为1
    data = pd.concat([ones,df],axis=1)#合并数据，根据列合并，表头为[ones, exam1, exam2]
    # print(data)
    return data.iloc[:,:-1].values#这个操作返回ndarray,不是矩阵

#读取标签
def get_y(df):
    """assume the last colum is the target"""
    return np.array(df.iloc[:,-1])#df.iloc[:,-1]是指出df的最后一列,转换成numpy的形式

def normalize_feature(df):
    """Applies function along input axis(default 0 ) of Dataframe."""
    return df.apply(lambda column:(column - column.mean()) / column.std())#特征缩放

X = get_X(data)#100x3
print(X.shape)

y = get_y(data)#100x1
print(y.shape)

#sigmoid函数，也是逻辑回归模型的假设函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))

#画出sigmoid函数
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(-10,10,step=0.01),sigmoid(np.arange(-10,10,step=0.01)))
ax.set_ylim((-0.1,1.1))
ax.set_xlabel('z',fontsize=18)
ax.set_ylabel('g(z)',fontsize=18)
ax.set_title('sigmoid funcion',fontsize=18)
# plt.show()

#代价函数(公式见吴恩达老师机器学习笔记的95页)
theta = np.zeros(3)#因为X的大小为100x3,所以theta为3x1

def cost(theta,X,y):
    return np.mean(-y*np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

#print(cost(theta,X,y))#初始的cost值

#梯度下降
def gradient(theta,X,y):
    """
    just 1 batch gradient
    """
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

# print(gradient(theta,X,y))

#使用scipy.optimize.minimize去寻找最优参数
import scipy.optimize as opt
res = opt.minimize(fun=cost,x0=theta,args=(X,y),method='Newton-CG',jac=gradient)
print(res)

#用训练集预测和验证
def predict(x,theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

final_theta = res.x
y_pred = predict(X,final_theta)

from sklearn.metrics import classification_report #这个包是评价报告
# print(classification_report(y,y_pred))

#寻找决策边界
print(res.x)#这是最终的theta值
#theta_0 + theta_1 * x + theta_2 * y = 0
#那么y = -(theta_0 / theta2) -(theta_1 / theta_2) * x
coef = -(res.x / res.x[2]) #得到直线的系数
#print(coef)

x = np.arange(130,step=0.1)
y = coef[0] + coef[1] * x

sns.set(context="notebook",style="ticks",font_scale=1.5)
sns.lmplot('exam1','exam2',hue='admitted',data=data,height=6,fit_reg=False,scatter_kws={"s":25})

plt.plot(x,y,'grey')
plt.xlim(0,130)
plt.ylim(0,130)
plt.title('Decision Boundary')
plt.show()
