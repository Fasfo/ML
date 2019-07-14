import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pickle
df=pd.read_csv("Home_price.csv",header=0)
df=pd.DataFrame(df)
print(df.area.to_numpy)
print("*************")
#plt.scatter(x.Area,x.Price)
#df=df1.ix[1:]
#print(df)
#plt.scatter(df[0],df[1])
reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)
#reg.predict([[3300]])
s=pd.read_csv("house_price.csv",index_col=0)
print("s={}".format(s))
print(s)
p=reg.predict(s)
print(p)
s['price']=p
print("s={}".format(s))
s.to_csv('prediction.csv')

with open('reg_pickle','wb') as f:
    pickle.dump(reg,f)