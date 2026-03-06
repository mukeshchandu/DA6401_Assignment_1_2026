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
        y_pred=sm(y_pred) #converting logits by applying softmax
        self.cache_soft=y_pred
        if self.loss_type=="cross_entropy":
            return -(1/y_true.shape[0])*np.sum(np.log(np.clip(y_pred,1e-15,1-1e-15))*y_true)
        else: #mean_squared_error
            return (1/(2*y_true.shape[0]))*np.sum(np.square(y_true-y_pred))

    def backward(self,y_true,y_pred):
        # always recompute softmax from logits directly so no dependency on cache
        p=sm(y_pred)
        if self.loss_type=="cross_entropy":
            return (p-y_true)/y_true.shape[0]
        else:
            # MSE gradient through softmax needs full jacobian
            # dL/dz_i = p_i * [(p_i - y_i) - sum_k(p_k*(p_k-y_k))] / N
            diff=p-y_true
            weighted=np.sum(p*diff,axis=1,keepdims=True)
            return p*(diff-weighted)/y_true.shape[0]
