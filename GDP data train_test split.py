import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("GDP.csv")
#plt.scatter(df.Service,df.GDP)
x=df[['Agriculture','Service','Manufacturing']]
y=df.GDP
print(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)