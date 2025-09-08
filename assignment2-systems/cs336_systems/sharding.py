import torch
import torch.distributed as dist

class ShardingOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        
        self.params = params
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self.optimizer = None
        self.param_to_rank = dict()
        self.num_param_per_rank = [0] * self.world_size

        super().__init__(params, defaults={})

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        for p, rank in self.param_to_rank.items():
            dist.broadcast(p.data, src=rank, async_op=False)


    def add_param_group(self, param_group: dict):
        local_rank_params_group = []

        for p in param_group['params']:
            insert_idx = self.num_param_per_rank.index(min(self.num_param_per_rank))
            self.num_param_per_rank[insert_idx] += p.numel()
            self.param_to_rank[p] = insert_idx

            if insert_idx == self.rank:
                local_rank_params_group.append(p)


        if local_rank_params_group:
            local_rank_params_group_dict = {}

            for k, v in param_group.items():
                if k == 'params':
                    local_rank_params_group_dict['params'] = local_rank_params_group
                else:
                    local_rank_params_group_dict[k] = v

            if self.optimizer is None:
                self.optimizer = self.optimizer_cls([local_rank_params_group_dict], **self.optimizer_kwargs)
            else:
                self.optimizer.add_param_group(local_rank_params_group)
        
        super().add_param_group(param_group)
    