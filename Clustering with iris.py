import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
dataset=load_iris()
#print(dir(dataset))
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
#print(df)
#-----------------------preprocessing-----------------------
scaler=MinMaxScaler()
scaler.fit(df)
df=scaler.transform(df)
df=pd.DataFrame(df)
#print(df)
#----------------------clustring------------------------
km=KMeans(n_clusters=3)
predicted=km.fit_predict(df)
df['predict']=predicted
#print(df)

#-------------------------visualization-------------------
df1=df[df.predict==0]
#print(df1)



#---------------------------deciding k
k_range=range(1,10)
sse=[]
for i in k_range:
    km=KMeans(n_clusters=i)
    km.fit(df.drop(['predict'],axis='columns'))
    sse.append(km.inertia_)
plt.plot(k_range,sse)
plt.xlabel('k')    
plt.ylabel('SSE')

'''x=df1.iloc[1:17]
df2=x.drop(['predict'],axis='columns')
print(km.predict(df2))'''
    