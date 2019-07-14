import pandas as pd
import numpy as np
import math
from sklearn import linear_model
df=pd.read_csv("multi_house_price.csv",index_col=0)
df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
reg=linear_model.LinearRegression()
#price=m1*area+m2*bedrooms+m3*age+b
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.intercept_)   #b
print(reg.coef_)    #[m1,m2,m3]
print(reg.predict([[30000,3,40]]))