import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
import sklearn.preprocessing as  mm
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates
data = scipy.io.loadmat('C:\\UWaterloo\Semester\Winter 2018\ECE 657A\Assignment 1\DataB.mat')
fea=data.get('fea')
df = pd.DataFrame(fea)
gnd=data.get('gnd')
gnd = pd.DataFrame(data=gnd,columns=['target'])
y = gnd['target'] 
#y=y.values()
#
df_Nan=df.isnull().any()#check is there any NAN in the DF
transpose_data=df.transpose();#cov operates row wise thats y wetranspose it
cov=np.cov(transpose_data)
eigenValues,eigenVectors=np.linalg.eig(cov)
idx = eigenValues.argsort()[::-1]
i=2
d=eigenVectors[:,i]
f=eigenValues[i]

from sklearn.decomposition import PCA
X_norm = (df - df.min())/(df.max() - df.min())
X_r1=X_norm.dot(eigenVectors)

pca = PCA(n_components=2)
principalComponentsPca1 = pca.fit_transform(X_norm)
PcaDf1 = pd.DataFrame(data = principalComponentsPca1)

finalDf = pd.concat([PcaDf1, gnd[['target']]],axis=1)
############################################################2
fig=plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
plt.scatter(finalDf[y==0][0], finalDf[y==0][1], label='Class 0', c='red')
plt.scatter(finalDf[y==1][0], finalDf[y==1][1], label='Class 1', c='blue')
plt.scatter(finalDf[y==2][0], finalDf[y==2][1], label='Class 2', c='lightgreen')
plt.scatter(finalDf[y==3][0], finalDf[y==3][1], label='Class 3', c='yellow')
plt.scatter(finalDf[y==4][0], finalDf[y==4][1], label='Class 4', c='pink')

plt.legend()
plt.show()
##############################################################3
pca1 = PCA(n_components=6)
principalComponentsPca2 = pca1.fit_transform(X_norm)
PcaDf2 = pd.DataFrame(data = principalComponentsPca2)
#
finalDf1 = pd.concat([PcaDf2.loc[:,4],PcaDf2.loc[:,5], gnd[['target']]],axis=1)

fig=plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 5', fontsize = 15)
ax.set_ylabel('Principal Component 6', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
plt.scatter(finalDf1[y==0][4], finalDf1[y==0][5], label='Class 0', c='red')
plt.scatter(finalDf1[y==1][4], finalDf1[y==1][5], label='Class 1', c='blue')
plt.scatter(finalDf1[y==2][4], finalDf1[y==2][5], label='Class 2', c='lightgreen')
plt.scatter(finalDf1[y==3][4], finalDf1[y==3][5], label='Class 3', c='yellow')
plt.scatter(finalDf1[y==4][4], finalDf1[y==4][5], label='Class 4', c='pink')

plt.legend()
plt.show()
##########################################################4

n1=(2, 4, 10, 30, 60, 200, 500,784)

principalComponentsPca={}
ratio={}

for i,x in zip(range(8),n1):
  
    pca1 = PCA(n_components=n1[i])
    principalComponentsPca[x]= pca1.fit_transform(X_norm)
    ratio[x]=pca1.explained_variance_ratio_.sum()
    
#test11 = principalComponentsPca.get(2)
#test1=pd.DataFrame(test)

#    PcaDf.append(n1) = pd.DataFrame(data = principalComponentsPca2)

from sklearn.cross_validation import train_test_split


key=(2, 4, 10, 30, 60, 200, 500,784)
a={}
accuracy={}
for i,k in zip(range(8),key):

    X=principalComponentsPca.get(key[i])
    y=gnd.values
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
    sc=mm.StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    from sklearn.metrics import accuracy_score
    a[i]=accuracy_score(y_test, y_pred,normalize=False)
    accuracy[i]=accuracy_score(y_test, y_pred,normalize=True)
    
a1=a
#r= [620-x for x in a]
a22={}
a2=620-a.get(2)
for key in range(8):
    a22[key] = 620-a.get(key)

plt.xlabel('Retained variance', fontsize=18)
plt.ylabel('Classification Error', fontsize=16)
plt.plot(list(ratio.values()),list(a22.values()))

##########################################################5
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
ldaComponents = lda.fit(X_norm, y).transform(X_norm)
ldaDf1 = pd.DataFrame(data = ldaComponents)
fig=plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('LDA 1', fontsize = 15)
ax.set_ylabel('LDA 2', fontsize = 15)
ax.set_title('2 component LDA', fontsize = 20)
plt.scatter(ldaDf1[y==0][0], ldaDf1[y==0][1], label='Class 0', c='red')
plt.scatter(ldaDf1[y==1][0], ldaDf1[y==1][1], label='Class 1', c='blue')
plt.scatter(ldaDf1[y==2][0], ldaDf1[y==2][1], label='Class 2', c='lightgreen')
plt.scatter(ldaDf1[y==3][0], ldaDf1[y==3][1], label='Class 3', c='yellow')
plt.scatter(ldaDf1[y==4][0], ldaDf1[y==4][1], label='Class 4', c='pink')

plt.legend()
plt.show() 

