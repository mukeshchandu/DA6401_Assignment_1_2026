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
        """Updates weights for all layers based on the chosen algorithm"""
        self.t += 1
        
        for layer in layers:
            # Apply L2 Regularization (Weight decay)
            if self.wd > 0:
                layer.grad_W += self.wd * layer.W
                
            if self.type == "sgd":
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b
                
            elif self.type == "momentum":
                beta = 0.9
                # Update velocities (you added v_w and v_b in your layer init!)
                layer.v_W = beta * layer.v_W + self.lr * layer.grad_W
                layer.v_b = beta * layer.v_b + self.lr * layer.grad_b
                
                # Update weights
                layer.W -= layer.v_W
                layer.b -= layer.v_b
                
            elif self.type == "nag":
                # Nesterov Accelerated Gradient
                beta = 0.9
                # Save current weights
                w_prev = layer.W.copy()
                b_prev = layer.b.copy()
                # Look-ahead step
                layer.W -= beta * layer.v_W
                layer.b -= beta * layer.v_b
                # Update velocity using gradient at look-ahead position
                layer.v_W = beta * layer.v_W + self.lr * layer.grad_W
                layer.v_b = beta * layer.v_b + self.lr * layer.grad_b
                # Restore and apply update
                layer.W = w_prev - layer.v_W
                layer.b = b_prev - layer.v_b

            elif self.type == "rmsprop":
                beta = 0.9
                epsilon = 1e-8
                layer.v_W = beta * layer.v_W + (1 - beta) * np.square(layer.grad_W)
                layer.v_b = beta * layer.v_b + (1 - beta) * np.square(layer.grad_b)
                layer.W -= self.lr * layer.grad_W / (np.sqrt(layer.v_W) + epsilon)
                layer.b -= self.lr * layer.grad_b / (np.sqrt(layer.v_b) + epsilon)