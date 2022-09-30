#正则化逻辑回归

"""
设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。对于这两次测试，你想决定是否芯片要被接受或抛弃。
为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，从其中你可以构建一个逻辑回归模型。
"""
import pandas as pd
import numpy as np
import seaborn as sns #seaborn 是在matplotlib的基础上进行了更高级的API封装
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report#这个包是评价报告

#数据可视化
path = './data/ex2data2.txt'
df = pd.read_csv(path, names=['test1', 'test2', 'accepted'])
print(df.head())#前5行的数据

sns.set(context="notebook",style="ticks",font_scale=1.5)
sns.lmplot('test1','test2',hue='accepted',data=df,
           height=6,
           fit_reg=False,
           scatter_kws={"s":50})
plt.title('Regularized Logistic Regression')
# plt.show()

#读取标签
def get_y(df):
    """assume the last column is the target"""
    return np.array(df.iloc[:,-1])


#特征映射，构造特征多项式
"""
polynomial expansion
for i in 0..i
  for p in 0..i:
    output x^(i-p) * y^p
参考吴恩达学习笔记的P113中的表达式h(x)
"""
def feature_mapping(x, y, power, as_ndarray=False):
#     """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)

x1 = np.array(df.test1)
x2 = np.array(df.test2)

data = feature_mapping(x1,x2,power=6)

print(data.shape)
print(data.head())

#sigmoid函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))

#不带正则项的代价函数
def cost(theta,X,y):
    return np.mean(-y * np.log(sigmoid(X@theta)) - (1-y) * np.log(1 - sigmoid(X @ theta)))#X @ theta与X.dot(theta)等价

#regularized cost (正则化代价函数)
theta = np.zeros(data.shape[1])
X = feature_mapping(x1, x2, power=6, as_ndarray=True)
print(X.shape)

y = get_y(df)
print(y.shape)

def regularized_cost(theta, X, y, l=1):
#     '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term#正则化代价函数

print(regularized_cost(theta,X,y,l=1))#因为我们设置theta为0，所以这个正则化代价函数与非正则化代价函数的值相同

#非正则化的梯度下降
def gradient(theta,X,y):
    """just 1 batch gradient"""
    return (1/len(X)) * X.T @ (sigmoid(X @ theta) - y)

#求正则化代价的梯度
def regularized_gradient(theta, X, y, l=1):
#     '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term

#拟合参数
import scipy.optimize as opt
print('init cost = {}'.format(regularized_cost(theta, X, y)))

res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)

#预测
def predict(x,theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

final_theta = res.x
y_pred = predict(X, final_theta)

print(classification_report(y, y_pred))

#使用不同的参数lambda(常数)画出决策边界
#我们找到所有满足  X×θ=0  的x

def draw_boundary(power, l):
#     """
#     power: polynomial power for mapped feature
#     l: lambda constant
#     """
    density = 1000
    threshhold = 2 * 10**-3

    final_theta = feature_mapped_logistic_regression(power, l)
    x, y = find_decision_boundary(density, power, final_theta, threshhold)

    df = pd.read_csv(path, names=['test1', 'test2', 'accepted'])
    sns.lmplot('test1', 'test2', hue='accepted', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})

    plt.scatter(x, y, c='R', s=10)
    plt.title('Decision boundary')
    plt.show()

def feature_mapped_logistic_regression(power, l):
#     """for drawing purpose only.. not a well generealize logistic regression
#     power: int
#         raise x1, x2 to polynomial power
#     l: int
#         lambda constant for regularization term
#     """
    df = pd.read_csv(path, names=['test1', 'test2', 'accepted'])
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    y = get_y(df)

    X = feature_mapping(x1, x2, power, as_ndarray=True)
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient)
    final_theta = res.x

    return final_theta

#寻找决策边界
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe

    inner_product = mapped_cord.values @ theta

    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision.f10, decision.f01
#寻找决策边界函数
# draw_boundary(power=6,l=1)#lambda=1
# draw_boundary(power=6,l=0)#lambda=0,没有正则化，过拟合了
draw_boundary(power=6,l=100)#lambda=100，欠拟合
