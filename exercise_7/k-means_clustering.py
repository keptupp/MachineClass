import scipy.io
import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
# from scipy.misc import imread

#Load example dataset
data = scipy.io.loadmat('C:\PythonCode\MachineClass\exercise_7\data\ex7data2.mat')
X = data['X']

#计算新的质心，并返回
count = 0
def compute_centroids(X,idx,K):
    m,n = X.shape #300,2
    #通过计算分配给每个质心的数据点的平均数返回新的质心
    centroid = np.zeros([K,n])#值全为0的张量，大小为Kxn

    for k in range(K):#k=0,..,K-1
        C = np.sum(idx==k)#求idx矩阵中，元素等于k的总数
        #将idx张量中，元素值等于k的赋值为True,不等于k的赋值为False，然后强制转换成整数的1,0.
        idx_k = (idx==k).astype(int)
        X_k = X * idx_k#idx_k中值为1的的元素*X，保留X中的元素

        mu = (1/C) * np.sum(X_k,axis=0)#求每一个簇的平均值
        centroid[k] = mu#更新每一个簇的平均值，存储到centroid中

    return centroid

#为数据集X中的每个示例返回最近的中心点。
def find_closest_centroids(X,centroids):
    idx = np.zeros([X.shape[0],1])#300X1

    K = centroids.shape[0]#中心点的个数
    m = X.shape[0] #元素的个数

    for i in range(m):
        c = -1 # index of closest centroid
        dist_min = np.inf #distance to nearest centroid

        for k in range(K):
            dist = distance.euclidean(centroids[k],X[i])
            if dist < dist_min:
                dist_min = dist
                c = k
        idx[i] = c

    return idx

def plot_k_means(X,idx,centroids_history,terminated=False):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0],X[:,1],marker='o',c=np.squeeze(idx),cmap='rainbow')

    if terminated:
        plt.plot(centroids_history[:,:,0],centroids_history[:,:,1],'x-',c='k')

    plt.show()

#Random initialization
def k_means_init_centroids(X,K):
    '''
    Return K random initial centroids to be
    used with the k-means clustering on the dataset K
    '''
    centroids = np.zeros([K,X.shape[1]])

    #Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    #Take the first K examples as centroids
    centroids = X[randidx[:K],:]

    return centroids

def run_k_means(X,initial_centroids,max_iters,plot_progress=False):
    m,n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros([m,1])

    centroids_history = np.zeros([max_iters+1,centroids.shape[0],centroids.shape[1]])
    centroids_history[0] = initial_centroids

    if plot_progress:
        print('Before clustering:')
        plot_k_means(X,idx,centroids_history)

    #Run k-means
    for i in range(max_iters):
        idx = find_closest_centroids(X,centroids)
        centroids = compute_centroids(X,idx,K)
        centroids_history[i+1] = centroids

    if plot_progress:
        print('Clustering after', max_iters,'interations of k-means:')
        plot_k_means(X,idx,centroids_history,terminated=True)

    return centroids,idx


initial_centroids = k_means_init_centroids(X,K=3)
centroid,idx = run_k_means(X,initial_centroids,max_iters=10,plot_progress=True)


#图片的压缩
import imageio
img = imageio.imread("C:\PythonCode\MachineClass\exercise_7\data\bird_small.png")
plt.imshow(img)
plt.show()

A = img
A = A / 255 #divide by 255 so all colour values are in range 0-1
X = np.reshape(A,[A.shape[0] * A.shape[1],3])

initial_centroids = k_means_init_centroids(X,K=16)
centroids,idx = run_k_means(X,initial_centroids,max_iters=10)

idx = idx.flatten()
idx = idx.astype(int)

#Recover compressed image
X_recovered = centroids[idx]

#X_recovered is now equivalent to our flattened X (16384 X 3), but in 16 colours.
# Reshape to turn it into a 128 x 128 image
X_recovered = np.reshape(X_recovered,img.shape)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.title('Original image in 24-bit colour')
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(X_recovered)
plt.title('Compressed image in 4-bit colour')
plt.show()

