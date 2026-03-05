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
        self.cache_soft = y_soft  # cache for backward if called next
        if self.loss_type == "cross_entropy":
            return -(1 / y_true.shape[0]) * np.sum(
                np.log(np.clip(y_soft, 1e-15, 1 - 1e-15)) * y_true
            )
        else:  # mean_squared_error
            return (1 / (2 * y_true.shape[0])) * np.sum(np.square(y_true - y_soft))

    def backward(self, y_true, y_pred):
        """
        Compute gradient of loss w.r.t. pre-softmax logits.
        Always recomputes softmax from y_pred (logits) directly —
        does NOT rely on cache_soft, so safe to call without forward().
        """
        y_soft = sm(y_pred)
        # For both CE and MSE with softmax output, the gradient
        # w.r.t. the logits simplifies to (softmax - y_true) / N
        return (y_soft - y_true) / y_true.shape[0]
