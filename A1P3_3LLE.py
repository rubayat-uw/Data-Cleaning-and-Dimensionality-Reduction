import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
import sklearn.preprocessing as  mm
import scipy.io
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = scipy.io.loadmat('C:\\UWaterloo\Semester\Winter 2018\ECE 657A\Assignment 1\DataB.mat')
fea=data.get('fea')
X = pd.DataFrame(fea)
gnd=data.get('gnd')
gnd = pd.DataFrame(data=gnd,columns=['target'])
y1 = gnd['target'] 
X_norm = (X - X.min())/(X.max() - X.min())
#############

from sklearn import manifold
from sklearn.cross_validation import train_test_split

lle = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=4)
lle.fit(X_norm)
lle_f = lle.transform(X_norm)
lle_df = pd.DataFrame(lle_f)

Xtrain=lle_df.values
y=gnd.values
accuracy_value={}
accuracy_rate={}
#randomly select test and train data
for i in range(100):
    X_train,X_test,y_train,y_test=train_test_split(Xtrain,y,test_size=0.30)
    sc=mm.StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.fit_transform(X_test)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    accuracy_value[i]=accuracy_score(y_test, y_pred,normalize=False)
    accuracy_rate[i]=accuracy_score(y_test, y_pred,normalize=True)
#get the error value
error_value={}
for key in range(100):
    error_value[key] = 620-accuracy_value.get(key)
plt.plot(list(accuracy_rate.values()))



data = [accuracy_rate[key] for k in range(100)]
print("Mean:\t"+ str(np.mean(data)) + "\nstd :\t" + str(np.std(data)))