import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digit=load_digits()
print(dir(digit))
print(len(digit.images))
#print(digit.data[0])
#plt.gray()

#plt.matshow(digit.images[0])
#plt.imshow()
#for i in range(5):
#    plt.matshow(digit.images[i])
    
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
X_train,X_test,Y_train,Y_test =train_test_split(digit.data,digit.target,test_size=0.21)
print(len(Y_test))

from sklearn.linear_model import LogisticRegression 
reg=LogisticRegression()
reg.fit(X_train,Y_train)
print(reg.score(X_test,Y_test))
#print(reg.predict(X_test))
print(reg.predict([digit.data[68]]))
print(digit.target[68])

#-----------------------------confusion matrix--------------------
y_predicted=reg.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_predicted)
#print(cm)

#seaborn library for visualization-----------------------
import seaborn as sn
#plt.figure()
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')

    
    