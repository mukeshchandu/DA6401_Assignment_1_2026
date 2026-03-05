"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from ann.activations import softmax as sm


class loss:
    def __init__(self, loss_type):
        self.loss_type = loss_type

    def forward(self, y_true, y_pred):
        """
        Compute loss value. Applies softmax to logits internally.
        """
        y_soft = sm(y_pred)
        self.cache_soft = y_soft
        if self.loss_type == "cross_entropy":
            return -(1 / y_true.shape[0]) * np.sum(
                np.log(np.clip(y_soft, 1e-15, 1 - 1e-15)) * y_true
            )
        else:  # mean_squared_error
            return (1 / (2 * y_true.shape[0])) * np.sum(np.square(y_true - y_soft))

    def backward(self, y_true, y_pred):
        """
        Compute gradient of loss w.r.t. pre-softmax logits.
        Always recomputes softmax from y_pred directly — safe without calling forward() first.

        For Cross-Entropy with Softmax:
            dL/dz = (softmax(z) - y) / N

        For MSE with Softmax (full Jacobian):
            dL/dz_i = p_i * [ (p_i - y_i) - sum_k(p_k * (p_k - y_k)) ] / N
        """
        N = y_true.shape[0]
        p = sm(y_pred)

        if self.loss_type == "cross_entropy":
            return (p - y_true) / N
        else:  # mean_squared_error — must account for softmax Jacobian
            diff = p - y_true                                    # (N, C)
            weighted = np.sum(p * diff, axis=1, keepdims=True)  # (N, 1)
            return p * (diff - weighted) / N
