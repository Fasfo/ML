import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
df=pd.read_csv("hr_salary.csv")
#plt.figure()
#plt.bar(df.salary,df.left,color='red')
#plt.figure()
#plt.bar(df.satisfaction_level,df.left)
#print(df.columns)
#plt.show()
x=df.drop('left',axis='columns') # left is column name in hr_salarry.csv file
#x=x.values
y=df.left
#print(x.columns)
#----------------------------------preprocessing---------------------------------.
#le=LabelEncoder()
#x.Department=le.fit_transform(x.Department)
#x.salary=le.fit_transform(x.salary)
dumies1=pd.get_dummies(x.Department)
dum1=dumies1.drop(['hr'],axis='columns')
#print(dum1)
dumies2=pd.get_dummies(x.salary)
dum2=dumies2.drop(['low'],axis='columns')
x=pd.concat([x,dumies1,dumies2],axis='columns')
x=x.drop(['Department','salary'],axis='columns')
#print(x)
#x=x.values
#print(x)
#ohe=OneHotEncoder(categorical_features=[-1])
#x=ohe.fit_transform(x).toarray()
#print(x)
#---------------------------training----------------------------
reg=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
reg.fit(x_train,y_train)

#x1=set(x_test['salary'])
#print(x1)
#print(len(x1))
#---------------------------testing----------------
print(reg.predict(x_test))
print(np.array(y_test))
print(reg.predict_proba(x_test))
print(reg.score(x_test,y_test))