"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from ann.activations import relu,sigmoid,tanh,relu_back,sigmoid_back,tanh_back,softmax,softmax_back
class Neurallayer:
    def __init__(self,in_dim,out_dim,activation="sigmoid",weight_init="random"):
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.activation=activation
        if weight_init=="xavier":
            k=np.sqrt(6.0/(in_dim+out_dim))
            self.W=np.random.uniform(-k,k,(in_dim,out_dim))
        else:self.W=np.random.randn(in_dim,out_dim)*0.01 if weight_init=="random" else np.zeros((in_dim,out_dim))
        self.b=np.zeros((1,out_dim))
        self.z=None
        self.a_prev=None
        self.grad_W=None
        self.grad_b=None
        self.v_W,self.v_b = np.zeros_like(self.W), np.zeros_like(self.b)
        self.m_W,self.m_b = np.zeros_like(self.W), np.zeros_like(self.b)
    def forward(self,a_prev):
        self.a_prev=a_prev
        self.z=a_prev@self.W+self.b
        return self.z if self.activation=="linear" else globals()[self.activation](self.z) 
    def backward(self,da_nextlayer):
        dz=da_nextlayer if self.activation=="linear" else da_nextlayer*globals()[f"{self.activation}_back"](self.z)
        self.grad_W=self.a_prev.T@dz
        self.grad_b =np.sum(dz,axis=0,keepdims=True)
        return dz@self.W.T



