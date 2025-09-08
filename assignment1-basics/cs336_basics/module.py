import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from torch import Tensor
from jaxtyping import Int, Float
from einops import rearrange, einsum
from .utils import softmax, silu
from .tokenizer import Tokenizer

class Linear(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_out: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty((self.d_out, self.d_in), device=device, dtype=dtype))
        self._initialize()

    def _initialize(self):
        std = np.sqrt(2/(self.d_in+self.d_out))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x:Float[torch.Tensor, '... d_in']) -> Float[torch.Tensor, '... d_out']:
        return x @ self.weight.transpose(-1, -2)
        # return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_emd = num_embeddings
        self.emd_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty((self.num_emd, self.emd_dim), device=device, dtype=dtype))
        
        self._initialize()

    def _initialize(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids:Int[torch.Tensor, '']) -> Float[torch.Tensor, '... d_model']:
        token_ids.to(dtype=torch.int64)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
       
    def forward(self, x: Float[torch.Tensor, '... d_model']) -> Float[torch.Tensor, '... d_model']:
        dtype = x.dtype
        x.to(torch.float32)
        x = x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        result = x * self.weight
        return result.to(dtype=dtype)
    

class SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
    
    def forward(self, x: Float[torch.Tensor, '... d_model']) -> Float[torch.Tensor, '... d_model']:
        return self.w2(silu(self.w1(x)))


class PositionwiseFeedForward(nn.Module): #SwiGLU
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x:Float[torch.Tensor, '... d_model']) -> Float[torch.Tensor, '... d_model']:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_frac = theta ** (-torch.arange(0, d_k, 2) / d_k)
        theta_table = einsum(torch.arange(max_seq_len), inv_frac, 'i,j->i j')

        cos_pos = theta_table.cos().repeat_interleave(repeats=2, dim=-1)
        sin_pos = theta_table.sin().repeat_interleave(repeats=2, dim=-1)

        self.register_buffer('cos_pos', cos_pos, persistent=False),
        self.register_buffer('sin_pos', sin_pos, persistent=False),

    def forward(
        self, 
        x: Float[torch.Tensor, '... seq d_model'], 
        token_positions: Int[torch.Tensor, '... seq']
    ) -> Float[torch.Tensor, '... seq d_model']:
        
        cos_pos = self.cos_pos[token_positions, :]
        sin_pos = self.sin_pos[token_positions, :]

        x_2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_2 = x_2.flatten(start_dim=-2)
        return x * cos_pos + x_2 * sin_pos


class RoPE(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        freqs = theta ** (-torch.arange(0, d_k, 2) / d_k)
        freqs = torch.outer(torch.arange(0, max_seq_len), freqs)
        self.freqs = torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        freqs = self.freqs[token_positions, ...]
        x = rearrange(x, '... (a b) -> ... a b', b = 2)
        x = torch.view_as_complex(x)
        result = torch.view_as_real(freqs * x).flatten(-2)
        return result

    

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, '... queries d_k'],
    K: Float[torch.Tensor, '... keys d_k'],
    V: Float[torch.Tensor, '... keys d_v'],
    mask: torch.Tensor | None = None
) -> Float[torch.Tensor, '... queires d_v']:

    d_k = Q.shape[-1]
    scale = d_k ** -0.5
    att_map = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") * scale
    if mask is not None:
        att_map = att_map.masked_fill(~mask, float('-inf'))
    att_map = softmax(att_map, -1)

    return einsum(att_map, V, "... query key, ... key d_v ->  ... query d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 use_rope: bool = False, 
                 theta: int | None = None, 
                 max_seq_len: int | None = None):
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * self.num_heads == self.d_model, "d_model must be divisible by num_heads"
    
        self.q_proj = Linear(d_in=self.d_model, d_out=self.d_model)
        self.k_proj = Linear(d_in=self.d_model, d_out=self.d_model)
        self.v_proj = Linear(d_in=self.d_model, d_out=self.d_model)
        self.output_proj = Linear(d_in=self.d_model, d_out=self.d_model)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)
        

    def forward(
        self, 
        x: Float[torch.Tensor, '... seq d_model'], 
        token_positions: Int[torch.Tensor, '... seq'] | None = None
    ) -> Float[torch.Tensor, '... seq d_model']:
        
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, '... seq_len (n d) -> ... n seq_len d', n=self.num_heads, d=self.head_dim)
        k = rearrange(k, '... seq_len (n d) -> ... n seq_len d', n=self.num_heads, d=self.head_dim)
        v = rearrange(v, '... seq_len (n d) -> ... n seq_len d', n=self.num_heads, d=self.head_dim)

        if self.use_rope is True:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        att_out = scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        att_out = rearrange(att_out, '... n seq_len d -> ... seq_len (n d)', n=self.num_heads, d=self.head_dim)
    
        return self.output_proj(att_out)
    
    def forward_version_2(self, x: torch.Tensor, token_positions: int | None =None) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(q.shape[:-1] + (self.num_heads, self.head_dim)).transpose(-2, -3)
        k = k.view(k.shape[:-1] + (self.num_heads, self.head_dim)).transpose(-2, -3)
        v = v.view(v.shape[:-1] + (self.num_heads, self.head_dim)).transpose(-2, -3)

        if token_positions is not None:
            assert self.use_rope is True
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        att_out = scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
        print(att_out.shape)
        att_out = att_out.transpose(-2, -3).contiguous().flatten(-2)

        return self.output_proj(att_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length
            ) 
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, 
            in_indices: Int[Tensor, " ... seq"],
            inference: bool = False
        ) -> Float[Tensor, " ... seq vocab_size"]:

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.ln_final(x)

        if inference is False:
            output = self.lm_head(x_norm)
        else:
            output = self.lm_head(x_norm[-1, None, :])

        return output

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    def generate_text(self, prompt: str, tokenizer: Tokenizer, max_token: int, temprature: float, top_p: float | None = None, top_k: float | None = None):
        assert temprature > 0, 'Invalid temperature'

        device = next(self.parameters()).device

        speicial_token_ids = set([tokenizer.encode(x)[0] for x in tokenizer.special_tokens])

        tokenize_ids = torch.tensor(tokenizer.encode(prompt), device=device)

        with torch.no_grad():
            for _ in range(max_token):
                logist = self.forward(tokenize_ids, inference=True)[-1]
                probs = softmax(logist / temprature, dim=-1)

                if top_p:
                    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = (cumulative_probs > top_p)

                    sorted_probs[mask] = 0.0
                    norm_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    sampled_indices = torch.multinomial(norm_probs, num_samples=1)

                    tokenize_ids = torch.cat([tokenize_ids, indices[sampled_indices]], dim=-1)
                
                if top_k:
                    topk_values, _ = torch.topk(
                        probs,
                        min(top_k, probs.size(-1)),
                    )
                    # Get the score of the kth item that we kept---items with lower scores should be masked.
                    threshold = topk_values[:, -1]
                    topk_mask = probs < threshold
                    probs.masked_fill(topk_mask, float("-inf"))
                    norm_probs = softmax(probs, dim=-1)
                    next_token_id = torch.multinomial(norm_probs, 1)
                    tokenize_ids = torch.cat([tokenize_ids, next_token_id], dim=-1)


                if tokenize_ids[-1].item() in speicial_token_ids:
                    break

        return tokenizer.decode(tokenize_ids.cpu().tolist())

# -------------------------------------------------------------

class TransformerBlockNoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=False, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerBlockNoRMS(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        # self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        # self.ln2 = RMSNorm(d_model=d_model)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class TransformerBlockPostNorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x
        

class TransformerBlockPostNorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x
    
class TransformerLMNoRMS(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            TransformerBlockNoRMS(
                d_model=d_model,
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length
            ) 
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, 
            in_indices: Int[Tensor, " batch_size sequence_length"],
            inference: bool = False
        ) -> Float[Tensor, " batchsize sequence_length d_model"]:

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.ln_final(x)

        if inference is False:
            output = self.lm_head(x_norm)
        else:
            output = self.lm_head(x_norm[-1, None, :])

        return output


class TransformerLMPostNorm(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            TransformerBlockPostNorm(
                d_model=d_model,
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length
            ) 
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, 
            in_indices: Int[Tensor, " batch_size sequence_length"],
            inference: bool = False
        ) -> Float[Tensor, " batchsize sequence_length d_model"]:

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.ln_final(x)

        if inference is False:
            output = self.lm_head(x_norm)
        else:
            output = self.lm_head(x_norm[-1, None, :])

        return output
    

class TransformerLMNoPE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            TransformerBlockNoPE(
                d_model=d_model,
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length
            ) 
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, 
            in_indices: Int[Tensor, " batch_size sequence_length"],
            inference: bool = False
        ) -> Float[Tensor, " batchsize sequence_length d_model"]:

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.ln_final(x)

        if inference is False:
            output = self.lm_head(x_norm)
        else:
            output = self.lm_head(x_norm[-1, None, :])

        return output
    
class TransformerBlockSiLU(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = SiLU(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLMSiLU(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            TransformerBlockSiLU(
                d_model=d_model,
                num_heads=num_heads, 
                d_ff=d_ff, 
                theta=rope_theta, 
                max_seq_len=context_length
            ) 
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, 
            in_indices: Int[Tensor, " batch_size sequence_length"],
            inference: bool = False
        ) -> Float[Tensor, " batchsize sequence_length d_model"]:

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)
        
        x_norm = self.ln_final(x)

        if inference is False:
            output = self.lm_head(x_norm)
        else:
            output = self.lm_head(x_norm[-1, None, :])

        return output

