#iris flower has two type of leave one is sepal and other has petal. based on height and widht of these leaves you can predict 
#type of iris flower

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
dataset=load_iris()
print(dir(dataset))
print(dataset.feature_names)
x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.2)


reg=LogisticRegression()
reg.fit(x_train,y_train)
print(reg.score(x_test,y_test))
print(reg.predict(x_test[0:5]))
print(y_test[0:5])