from sklearn.preprocessing import Imputer
import sklearn.preprocessing as  mm
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = scipy.io.loadmat('C:\\UWaterloo\Semester\Winter 2018\ECE 657A\Assignment 1\DataA.mat')
half_count = len(data) / 2
s=data.get('fea')
df = pd.DataFrame(s)

original_data= df.dropna(thresh=half_count,axis=1) 
#convert NAN into mean
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
new_data1 = imp.fit_transform(original_data)
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(original_data)# make new columns indicating what will be imputed
data_with_imputed_values_df = pd.DataFrame(data_with_imputed_values)
values=data_with_imputed_values_df.values
##plt hist of all attributes
#i=0
#for i in data_with_imputed_values_df:
#    plt.title('f model: T=%d' %i)
#    plt.hist(data_with_imputed_values_df[i], color='g')
#    
#    plt.figure()
#    plt.show()
#data_with_imputed_values_df.hist(0)

#min max
scaler = mm.MinMaxScaler(feature_range=(0, 1))
scalar=scaler.fit(values)
norm=scaler.transform(values)
min_max_norm = pd.DataFrame(norm)
min_max_norm
#z-score
scaler2=mm.StandardScaler()
scaler2=scaler2.fit(values)
z_score = scaler2.transform(values)
z_score = pd.DataFrame(z_score)
z_score
#z_score.hist(24)

plt.title('feature 9 plot')
plt.grid()
plt.hist(data_with_imputed_values_df[9])
plt.figure()
plt.show()

min_max_norm.hist(9)

plt.title('feature 24 plot')
plt.hist(data_with_imputed_values_df[24])
plt.grid()
plt.figure()
plt.show()

z_score.hist(24)