import pandas as pd
import numpy as np
import math
#df=pd.DataFrame(list(zip([2300,3000,3200,3600,4000],[3,4,np.nan,3,5],
 #       [20,15,18,30,8],[550000,565000,610000,595000,760000])),columns=['area','bedrooms','age','price'])
#df.to_csv("multi_house_price.csv")
#print(df.bedrooms.median())



#lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks'] 
#df = pd.DataFrame(lst, index =['a', 'b', 'c', 'd', 'e', 'f', 'g'], columns =['Names']) 
  
#lst = [['tom', 25], ['krish', 30], 
 #      ['nick', 26], ['juli', 22]] 
    
#df = pd.DataFrame(lst, columns =['Name', 'Age']) 
 
 #---------------------Dummy variable and hot encoding_______________________
 
df=pd.DataFrame(list(zip(['up','mp','up','hp','mp'],[2300,3000,3200,3600,4000],[3,4,np.nan,3,5],
        [20,15,18,30,8],[550000,565000,610000,595000,760000])),columns=['state','area','bedrooms','age','price'])
dummy=pd.get_dummies(df.state)
df1=pd.concat([df,dummy],axis='columns')
df2=df1.drop(['state','hp'],axis='columns')
x=df2.drop('price',axis='columns')
x.bedrooms=x.bedrooms.fillna(x.bedrooms.mean())
from sklearn import linear_model
import pickle
reg=linear_model.LinearRegression()
reg.fit(x,df2.price)
print(reg.predict([[3000,4,15,1,0]]))
#reg.score(x,y)
with open('dummies_mlv','wb') as f:
    pickle.dump(reg,f)
    
    
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le=LabelEncoder ()
dfle=df
dfle.state=le.fit_transform(dfle.state)
print(dfle)

