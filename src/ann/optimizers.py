"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class Optimizer:
    def __init__(self, optimizer_type, lr, weight_decay=0.0):
        self.type = optimizer_type
        self.lr = lr
        self.wd = weight_decay
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if self.wd > 0:
                layer.grad_W += self.wd * layer.W
            if self.type == "sgd":
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b
            elif self.type == "momentum":
                b = 0.9
                layer.v_W = b*layer.v_W + self.lr*layer.grad_W
                layer.v_b = b*layer.v_b + self.lr*layer.grad_b
                layer.W -= layer.v_W
                layer.b -= layer.v_b
            elif self.type == "nag":
                b = 0.9
                w_prev, b_prev = layer.W.copy(), layer.b.copy()
                layer.W -= b*layer.v_W
                layer.b -= b*layer.v_b
                layer.v_W = b*layer.v_W + self.lr*layer.grad_W
                layer.v_b = b*layer.v_b + self.lr*layer.grad_b
                layer.W = w_prev - layer.v_W
                layer.b = b_prev - layer.v_b
            elif self.type == "rmsprop":
                b, eps = 0.9, 1e-8
                layer.v_W = b*layer.v_W + (1-b)*np.square(layer.grad_W)
                layer.v_b = b*layer.v_b + (1-b)*np.square(layer.grad_b)
                layer.W -= self.lr*layer.grad_W/(np.sqrt(layer.v_W)+eps)
                layer.b -= self.lr*layer.grad_b/(np.sqrt(layer.v_b)+eps)