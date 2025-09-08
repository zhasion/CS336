import torch
import torch.nn as nn
import torch.distributed as dist


class Bucket():
    def __init__(self):
        self.params :list[nn.Parameter] = []
        self.num_params: int = 0
        self.num_params_ready: int = 0
    

class DDPOverlapBucketed(nn.Module):
    def __init__(self, model: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = model
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()

        self.num_bucket_parameters = self.bucket_size_mb * 2 ** 20 // next(self.module.parameters()).dtype.itemsize
        self.bucket = []
        self._init_bucket()

        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        
        self.handle_list = []
        self._register_hook()


    def _init_bucket(self):
        bucket_idx = 0
        bucket = Bucket()
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue

            if bucket.num_params + p.numel() >= self.num_bucket_parameters:
                self.bucket.append(bucket)
                bucket = Bucket()
                bucket_idx += 1

            p.bucket_idx = bucket_idx
            bucket.params.append(p)
            bucket.num_params += p.numel()


        if bucket.num_params != 0:
            self.bucket.append(bucket)
                 

    def _register_hook(self):

        def hook(param: torch.Tensor):
            bucket = self.bucket[param.bucket_idx]
            bucket.num_params_ready += param.numel()
            if bucket.num_params_ready == bucket.num_params:
                params_with_grad = [p for p in bucket.params if p.grad is not None]
                if not params_with_grad:
                    return
                grads = [p.grad for p in params_with_grad]
                flatted_tensor = torch._utils._flatten_dense_tensors(grads)
                handle = dist.all_reduce(flatted_tensor, async_op=True)
                self.handle_list.append((handle, flatted_tensor, params_with_grad, grads))
                bucket.num_params_ready = 0

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


    def finish_gradient_synchronization(self):
        for (handle, flatted_tensor, params_with_grad, grads) in self.handle_list:
            handle.wait()

            flatted_tensor.div_(self.world_size)
            unflatten_tensor = torch._utils._unflatten_dense_tensors(flatted_tensor, grads)

            for p, new_grad in zip(params_with_grad, unflatten_tensor):
                p.grad = new_grad
        
        self.handle_list.clear()