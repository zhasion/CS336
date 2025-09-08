import torch, transformers, os
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input

torch.manual_seed(42)
device = "cuda"

# 1. 造一批假数据，和 Qwen2.5-Math 维度一致
B, T, H, D = 4, 2048, 32, 128   # 对应 7B
q = torch.randn(B*T, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn(B*T, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn(B*T, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)

# 2. 构造 seqlens（假设每个样本 512 token）
seqlens = [512] * B
cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(seqlens), 0).tolist(), dtype=torch.int32, device=device)
max_seqlen = max(seqlens)

# 3. 跑一次 forward / backward
with torch.autograd.set_detect_anomaly(True):
    out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, causal=True)
    loss = out.sum()
    loss.backward()
    
print("flash_attn_varlen_func OK")