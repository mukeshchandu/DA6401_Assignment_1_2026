"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np
def relu(x):
    return np.maximum(0,x)
def sigmoid(x):
    return 1/(1+(np.exp(-x)))
def tanh(x):
    return np.tanh(x)
def softmax(x):
    x=x-np.max(x,axis=-1,keepdims=True)
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
def relu_back(x):
    y=x.copy()
    y[x<=0]=0
    y[x>0]=1
    return y
def sigmoid_back(x):
    x=sigmoid(x)
    return x*(1-x)
def tanh_back(x):
    return (1-tanh(x)**2)
def softmax_back(x):
    return 1