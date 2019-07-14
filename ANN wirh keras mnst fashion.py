import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print(type(x_train))
#plt.matshow(x_train[1])
#print(y_train[1])
x_train=x_train/255         #normalize
x_test=x_test/255            #normalize
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
model=Sequential()
model.add(Flatten(input_shape=[28,28]))   # Flatten is used to convert 2D array into 1D array
model.add(Dense(20, activation='relu' ))   # Hidden layer has 20 node and activation function is Relu
model.add(Dense(10,activation='softmax'))  # output layer has 10 node and activation function is softmax (is has 10 type of item)
#print(model.summary())
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=["accuracy"])
model.fit(x_train,y_train)
yp=model.predict(x_test)
#print("-------------------")
#print(model.predict(x_test[0:1]))
'''print("-------------------")
print(yp[1])
print("-------------------")
x=np.argmax(yp[1])
print(x)
plt.imshow(x_test[1])'''
print(model.evaluate(x_test,y_test)) # [a, b] first parameter is loss, 2nd is accuracy