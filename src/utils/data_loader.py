"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist,fashion_mnist
from sklearn.model_selection import train_test_split as tst
def hot(y,n=10):
    x=[]
    for k in y:
        k+=1
        x.append([0]*(k-1)+[1]+[0]*(n-k))
    return np.array(x)
def load(name="mnist",val=0.1):
    (x_train,y_train),(x_test,y_test)=mnist.load_data() if name=="mnist" else fashion_mnist.load_data()
    x_train=x_train.reshape(x_train.shape[0],-1).astype(np.float32)/255.0
    x_test=x_test.reshape(x_test.shape[0],-1).astype(np.float32)/255.0
    y_train=hot(y_train)
    y_test=hot(y_test)
    x_train,x_val,y_train,y_val=tst(x_train,y_train,test_size=val,random_state=42)
    return x_train,y_train,x_val,y_val,x_test,y_test
