import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
dataset=load_digits()
#print(dir(dadaset))
#print(dataset.target)
df=pd.DataFrame(dataset.data)
df['target']=dataset.target
print(df.head(20))
#-----------------visualization----------------
x0=df[df.target==0]
x1=df[df.target==1]
x2=df[df.target==2]

plt.scatter(x0[2],x0[3],color='green',marker='*')
plt.scatter(x1[2],x1[3],color='red',marker='+')
plt.scatter(x2[2],x2[3])

#------------------classification
df1=df.drop('target',axis='columns')
x_train,x_test,y_train,y_test=train_test_split(df1,df.target,test_size=0.2)
#print(len(x_test))
model=SVC(kernel='linear')
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(model.predict(x_test[0:20]))
print(y_test[0:20])