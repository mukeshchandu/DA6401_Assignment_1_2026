"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from ann.activations import softmax as sm
class loss:
    def __init__(self,loss_type):
        self.loss_type=loss_type
    def forward(self,y_true,y_pred):
        y_pred=sm(y_pred)
        return -(1/y_true.shape[0])*np.sum(np.log(np.clip(y_pred,1e-15,1-1e-15))*y_true) if self.loss_type=="cross_entropy" else (1/(2*y_true.shape[0]))*np.sum(np.square(y_true-y_pred))
    def backward(self,y_true,y_pred):
        y_pred=sm(y_pred)
        return (y_pred-y_true)/(y_true.shape[0]) if self.loss_type=="cross_entropy" else y_pred*(y_pred-y_true-np.sum(y_pred*(y_pred-y_true),axis=1,keepdims=True))/y_true.shape[0]
    