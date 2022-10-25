#用scikit-learn来实现K-means.
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io
#cast to float, you need to do this otherwise the color would be weird after clusting
pic = io.imread('data/bird_small.png') / 255.
# io.imshow(pic)
# plt.show()
print(pic.shape)

#serialize data
data = pic.reshape(128*128,3)#16384*3
print(data.shape)

from sklearn.cluster import KMeans #导入Kmeans库

if __name__=='__main__':
   model = KMeans(n_clusters=16, n_init=10, n_jobs=-1)
   model.fit(data)

   centroids = model.cluster_centers_
   C = model.predict(data)
   compressed_pic = centroids[C].reshape((128,128,3))
   plt.figure(figsize=(10,6))
   plt.subplot(1,2,1)
   plt.title('Original image in 24-bit colour')
   plt.imshow(pic)
   plt.subplot(1,2,2)
   plt.imshow(compressed_pic)
   plt.title('Compressed image in 4-bit colour')
   plt.show()


