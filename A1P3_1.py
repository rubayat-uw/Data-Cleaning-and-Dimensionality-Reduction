import scipy.io
import struct
import numpy as np
import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt
digits= scipy.io.loadmat('DataB.mat')
train_data = digits.get('fea')
train_data=pd.DataFrame(train_data)

train_label = digits.get('gnd')
train_label = pd.DataFrame(data=train_label,columns=['target'])
train_label = train_label['target']

X = train_data[train_label==3]
X=X.reset_index(drop=True)

Y = manifold.LocallyLinearEmbedding(n_neighbors = 5, n_components =2). fit_transform(X)

plt.scatter(Y[ :,0], Y[ :,1])

def find_landmarks(Y, n, m):
    xr = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), n)
    yr = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), m)
    xg, yg = np.meshgrid(xr,yr)
    idx= [0]*(n*m)
    
    for i, x, y in zip(range(n*m), xg.flatten(), yg.flatten()):
        idx[i] = int(np.sum(np.abs(Y-np.array([x,y]))**2,axis = -1).argmin())
    return idx

landmarks =find_landmarks(Y,5,5)
plt.scatter(Y[ :,0], Y[ :,1])
plt.scatter(Y[landmarks,0], Y[landmarks,1])
plt.title("Locally Linear Embedding Component")
plt.xlabel("1st Component of LLE")
plt.ylabel("2nd Component of LLE")

#Test=X[landmarks[1]]
#AA=np.reshape(X[landmarks[1]], (28,28))
#X=X.reset_index(drop=True)
X=X.transpose() 

fig=plt.figure(figsize=(15,15))
for i in range(len(landmarks)):
    ax= fig.add_subplot(5,5,i+1)
    imgplot =ax.imshow(np.reshape(X[landmarks[i]], (28,28)), cmap=plt.cm.get_cmap("Greys"))
    imgplot.set_interpolation("nearest")
plt.show()
