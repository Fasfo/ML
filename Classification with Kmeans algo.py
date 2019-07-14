import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df=pd.DataFrame(list(zip(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w'],
                         [27,29,29,28,42,39,41,38,36,35,37,26,27,28,29,32,40,41,43,39,41,39],
                         [70,90,61,60,150,155,160,162,156,130,137,45,48,51,49.5,53,65,63,64,80,82,58])),
                        columns=['name','age','salary'])
#print(df)
#plt.scatter(df.age,df.salary)
'''km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['age','salary']])
#print(y_predicted)

df['predict']=y_predicted
#print(df.head())

df1=df[df.predict==0]
df2=df[df.predict==1]
df3=df[df.predict==2]

plt.scatter(df1.age,df1.salary,color='green')
plt.scatter(df2.age,df2.salary,color='red')
plt.scatter(df3.age,df3.salary,color='black')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()'''

#Due to scaling factor some of the point are wrong classified to overcome that lets do the scaling

scaler=MinMaxScaler()
scaler.fit(df[['salary']])
df['salary']=scaler.transform(df[['salary']])
scaler.fit(df[['age']])
df['age']=scaler.transform(df[['age']])
#print(df)
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['age','salary']])
#print(y_predicted)
center=km.cluster_centers_
df['predict']=y_predicted
df1=df[df.predict==0]
df2=df[df.predict==1]
df3=df[df.predict==2]

plt.scatter(df1.age,df1.salary,color='green')
plt.scatter(df2.age,df2.salary,color='red')
plt.scatter(df3.age,df3.salary,color='black')
plt.scatter(center[:,0],center[:,1],marker='*',label='center')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()


#Elbow techniques to decide the value of k
k_range=range(1,10)
sse=[]
for i in k_range:
    km=KMeans(n_clusters=i)
    km.fit(df[['age','salary']])
    sse.append(km.inertia_)               #km.inertia gives the sse
plt.plot(k_range,sse)
plt.xlabel('k')
plt.ylabel('sum of squared error')    

