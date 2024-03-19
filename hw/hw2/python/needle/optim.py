"""Optimization module"""
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
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # self.clip_grad_norm()
        for p in self.params:
            # It is necessary to check if a parameter has a gradient since it
            # may not constitute the computational graph. This happens when
            # multiplying a parameter with 0, which is simply weeded out in
            # code using branches.
            if p.grad is None:
                continue
            p_id = id(p)
            grad = (1. - self.momentum) * (p.grad.data + self.weight_decay * p.data) + \
                   self.momentum * self.u.get(p_id, 0.)
            print(grad.shape)
            p.data -= self.lr * grad
            self.u[p_id] = grad
        # for p in self.params:
        #     if p is None or p.grad is None:
        #         continue
        #     if self.weight_decay > 0:
        #         grad = p.grad.data + self.weight_decay * p.data
        #     else:
        #         grad = p.grad.data
        #     # grad = p.grad.data + self.weight_decay * p.data
        #     p_id = id(p)
        #     if p_id not in self.u:
        #         self.u[p_id] = ndl.init.zeros(*grad.shape, device=grad.device, dtype=grad.dtype)
        #     self.u[p_id] = self.momentum * self.u[p_id] + (1 - self.momentum) * grad
        #     p.data = p.data - self.lr * self.u[p_id]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        # clip_coef = max_norm / (total_norm + 1e-6)
        # clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        # for p in self.params:
        #     p.grad = p.grad.detach() * clip_coef_clamped
        # total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        # clip_coef = max_norm / (total_norm + 1e-6)
        # clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        # for p in self.params:
        #     p.grad = p.grad.detach() * clip_coef_clamped
        pass

        # for p in self.params:
        #     if p.grad is not None:
        #         grad_norm = np.linalg.norm(p.grad)
        #         if grad_norm > max_norm:
        #             p.grad *= max_norm / grad_norm
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

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
