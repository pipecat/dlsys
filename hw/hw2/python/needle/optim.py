"""Optimization module"""
from collections import defaultdict
from email.policy import default

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            if self.weight_decay > 0:
                grad = p.grad.data + self.weight_decay * p.data
            else:
                grad = p.grad.data
            p_id = id(p)
            if p_id not in self.u:
                self.u[p_id] = ndl.init.zeros(*grad.shape, device=grad.device, dtype=grad.dtype)
            self.u[p_id] = self.momentum * self.u[p_id] + (1 - self.momentum) * grad
            p.data = p.data - self.lr * self.u[p_id]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION

        for p in self.params:
            if p.grad is not None:
                grad_norm = np.linalg.norm(p.grad)
                if grad_norm > max_norm:
                    p.grad *= max_norm / grad_norm
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            if self.weight_decay > 0:
                grad = p.grad.data + self.weight_decay * p.data
            else:
                grad = p.grad.data
            p_id = id(p)
            if p_id not in self.m:
                self.m[p_id] = ndl.init.zeros(*grad.shape, device=grad.device, dtype=grad.dtype)
                self.v[p_id] = ndl.init.zeros(*grad.shape, device=grad.device, dtype=grad.dtype)
            self.m[p_id] = self.beta1 * self.m[p_id] + (1 - self.beta1) * grad
            self.v[p_id] = self.beta2 * self.v[p_id] + (1 - self.beta2) * grad ** 2
            p.data = p.data - self.lr * self.m[p_id] / (1 - self.beta1 ** self.t) / ((self.v[p_id] / (1 - self.beta2 ** self.t)) ** 0.5 + self.eps)

        ### END YOUR SOLUTION
