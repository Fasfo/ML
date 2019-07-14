import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
dataset=load_iris()
t=dataset.target

'''df1=pd.DataFrame(t)
df1.columns=['target']
print(df1)'''
#---------other way

df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
#df2=pd.concat([df,df1],axis='columns')
df['target']=t                                                         # add columns name target
df['flower name']=df.target.apply(lambda x:dataset.target_names[x])    #add columns flower name
#print(df.tail(20))
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='*')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='yellow')

#-------------------------------------------------------
x=df.drop(['target','flower name'],axis='columns')
#print(x.head())
x_train,x_test,y_train,y_test=train_test_split(x,dataset.target,test_size=0.2)
#print(len(x_test))
model=SVC(C=10,kernel='poly')
model.fit(x_train,y_train)
print(model.score(x_test,y_test))