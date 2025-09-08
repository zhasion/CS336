import torch
import math
from collections.abc import Callable
from typing import Optional

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError('Invalid Learing Rate!')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get iteration number from the state, or initial value.
                t = self.state[p].get('t', 0)

                # Get the gradient of loss with respect to p.
                grad = p.grad.data

                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad

                # Increment iteration number.
                self.state[p]['t'] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params: list|dict, 
                 lr: float, 
                 weight_decay: float = 1e-2, 
                 betas: tuple[float, float] = (0.9, 0.99), 
                 eps: float = 1e-8):
        if lr < 0:
            raise ValueError('Invalid learning rate: {lr}')
        if eps < 0:
            raise ValueError('Invalid eps: {eps}')
        if weight_decay < 0:
            raise ValueError('Invalid weight decay: {weight_decay}')
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError('Invalid betes: {betas}')
        
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps}
        super().__init__(params, defaults)

    
    def step(self, closure:Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the gradient of loss with respect to p.
                grad = p.grad.data

                # Get other params from the state, or initial value.
                t = self.state[p].get('t', 1)
                m = self.state[p].get('m', torch.zeros_like(p))
                v = self.state[p].get('v', torch.zeros_like(p))

                # Compute the momentum estimate.
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                
                # Bias correction.
                m_hat = m / (1 - betas[0] ** t)
                v_hat = v / (1 - betas[1] ** t)

                # Update the parameters.
                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)

                # Apply the weight decay.
                p.data -= lr * weight_decay * p.data

                # Update the momentum estimate.
                self.state[p]['m'] = m
                self.state[p]['v'] = v
                self.state[p]['t'] = t + 1

        return loss

if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)

    for t in range(10):
        opt.zero_grad()
        loss = weights.pow(2).mean()
        print(f'{(loss.cpu().item()):.5f}')
        loss.backward()
        opt.step()