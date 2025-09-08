import torch
import torch.nn as nn
import torch.distributed as dist

class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handle = []
        
        def hook(params: torch.Tensor):
            hd = dist.all_reduce(params.grad, async_op=True)
            self.handle.append((hd, params))

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, params in self.handle:
            handle.wait()
            params.grad.div_(self.world_size)
        self.handle.clear()

