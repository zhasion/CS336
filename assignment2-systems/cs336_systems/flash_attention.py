import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od, 
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0)
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    max_score = tl.full((Q_TILE_SIZE, 1), value=float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_tile_index in range(0, N_KEYS, K_TILE_SIZE):
        # Load k, v block.
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # Compute the attention score.
        attention_score = tl.dot(q, k.T) * scale

        # Use causal mask if necessary.
        if is_causal:
            q_star_idx = query_tile_index * Q_TILE_SIZE
            k_star_idx = key_tile_index

            q_pos = q_star_idx + tl.arange(0, Q_TILE_SIZE)[:, None]   # (Q_TILE_SIZE, 1)
            k_pos = k_star_idx + tl.arange(0, K_TILE_SIZE)[None, :]   # (1,   K_TILE_SIZE)

            causal_mask = q_pos >= k_pos
            attention_score = tl.where(causal_mask, attention_score, float('-inf'))
        
        # Find the max number in the block
        current_max_score = tl.maximum(max_score, tl.max(attention_score, axis=-1, keep_dims=True))
        exp_score = tl.exp(attention_score - current_max_score)

        # Online softmax, update previous data based on current max_score.
        sum_exp = sum_exp * tl.exp(max_score - current_max_score) + tl.sum(exp_score, axis=-1, keep_dims=True)
        output = output * tl.exp(max_score - current_max_score) + tl.dot(exp_score.to(v.dtype), v)

        # Update max score.
        max_score = current_max_score

        # Update block pointers for next iteration
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(O_block_ptr, (output / sum_exp).to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, (tl.log(sum_exp) + max_score).to(L_block_ptr.type.element_ty))


@triton.jit
def flash_bwd_kernel_kv(
    Q, K, V, 
    output, grad_output, L, D,
    DQ, DK, DV,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    stride_gb, stride_gs, stride_gd,
    stride_lb, stride_ls,
    stride_db, stride_ds,
    stride_dqb, stride_dqs, stride_dqd,
    stride_dkb, stride_dks, stride_dkd,
    stride_dvb, stride_dvs, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base = Q + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qs, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    DO_block_ptr = tl.make_block_ptr(
        base = grad_output + batch_index * stride_gb,
        shape=(N_QUERIES, DIM),
        strides=(stride_gs, stride_gd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        base = L + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_ls, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        base = D + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_ds, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0)
    )

    # DQ_block_ptr = tl.make_block_ptr(
    #     base = DQ + batch_index * stride_dqb,
    #     shape=(N_QUERIES, DIM),
    #     strides=(stride_dqs, stride_dqd),
    #     offsets=(0, 0),
    #     block_shape=(Q_TILE_SIZE, DIM),
    #     order=(1, 0)
    # )

    K_block_ptr = tl.make_block_ptr(
        base = K + batch_index * stride_kb,
        shape=(N_KEYS, DIM),
        strides=(stride_ks, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base = V + batch_index * stride_vb,
        shape=(N_KEYS, DIM),
        strides=(stride_vs, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # O_block_ptr = tl.make_block_ptr(
    #     base = output + batch_index * stride_ob,
    #     shape=(N_KEYS, DIM),
    #     strides=(stride_os, stride_od),
    #     offsets=(key_tile_index * K_TILE_SIZE, 0),
    #     block_shape=(K_TILE_SIZE, DIM),
    #     order=(1, 0)
    # )

    DK_block_ptr = tl.make_block_ptr(
        base = DK + batch_index * stride_dkb,
        shape=(N_KEYS, DIM),
        strides=(stride_dks, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    DV_block_ptr = tl.make_block_ptr(
        base = DV + batch_index * stride_dvb,
        shape=(N_KEYS, DIM),
        strides=(stride_dvs, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
    # o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option='zero')


    # dq = tl.zeros_like(q)
    dk = tl.zeros(k.shape, dtype=tl.float32)
    dv = tl.zeros(v.shape, dtype=tl.float32)

    for query_tile_index in range(0, N_QUERIES, Q_TILE_SIZE):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
        l = tl.load(L_block_ptr, boundary_check=(0, 1), padding_option='zero')
        d = tl.load(D_block_ptr, boundary_check=(0, 1), padding_option='zero')
        do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option='zero')


        s = tl.dot(q, k.T) * scale

        if is_causal:
            q_star_idx = query_tile_index
            k_star_idx = key_tile_index * K_TILE_SIZE

            q_pos = q_star_idx + tl.arange(0, Q_TILE_SIZE)[:, None]   # (Q_TILE_SIZE, 1)
            k_pos = k_star_idx + tl.arange(0, K_TILE_SIZE)[None, :]   # (1,   K_TILE_SIZE)

            causal_mask = q_pos >= k_pos
            s = tl.where(causal_mask, s, float('-inf'))

        p = tl.exp(s - l)

        dv += tl.dot(p.to(dtype=q.dtype).T, do)
        dp = tl.dot(do, v.T)
        ds = p * (dp - d)
        # dq += tl.dot(ds, k) * scale
        dk += tl.dot(ds.to(dtype=q.dtype).T, q) * scale

        # Update block pointers for next iteration
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, 0))
        DO_block_ptr = DO_block_ptr.advance((Q_TILE_SIZE, 0))

    # tl.store(DQ_block_ptr, dq.to(Q_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(DK_block_ptr, dk.to(K_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(DV_block_ptr, dv.to(V_block_ptr.type.element_ty), boundary_check=(0, 1))



@triton.jit
def flash_bwd_kernel_q(
    Q, K, V, 
    output, grad_output, L, D,
    DQ, DK, DV,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    stride_gb, stride_gs, stride_gd,
    stride_lb, stride_ls,
    stride_db, stride_ds,
    stride_dqb, stride_dqs, stride_dqd,
    stride_dkb, stride_dks, stride_dkd,
    stride_dvb, stride_dvs, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base = Q + batch_index * stride_qb,
        shape=(N_QUERIES, DIM),
        strides=(stride_qs, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    DO_block_ptr = tl.make_block_ptr(
        base = grad_output + batch_index * stride_gb,
        shape=(N_QUERIES, DIM),
        strides=(stride_gs, stride_gd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        base = L + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_ls, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        base = D + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_ds, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0)
    )

    DQ_block_ptr = tl.make_block_ptr(
        base = DQ + batch_index * stride_dqb,
        shape=(N_QUERIES, DIM),
        strides=(stride_dqs, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, DIM),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base = K + batch_index * stride_kb,
        shape=(N_KEYS, DIM),
        strides=(stride_ks, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base = V + batch_index * stride_vb,
        shape=(N_KEYS, DIM),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, DIM),
        order=(1, 0)
    )

    # O_block_ptr = tl.make_block_ptr(
    #     base = output + batch_index * stride_ob,
    #     shape=(N_KEYS, DIM),
    #     strides=(stride_os, stride_od),
    #     offsets=(0, 0),
    #     block_shape=(K_TILE_SIZE, DIM),
    #     order=(1, 0)
    # )

    # DK_block_ptr = tl.make_block_ptr(
    #     base = DK + batch_index * stride_dkb,
    #     shape=(N_KEYS, DIM),
    #     strides=(stride_dks, stride_dkd),
    #     offsets=(0, 0),
    #     block_shape=(K_TILE_SIZE, DIM),
    #     order=(1, 0)
    # )

    # DV_block_ptr = tl.make_block_ptr(
    #     base = DV + batch_index * stride_dvb,
    #     shape=(N_KEYS, DIM),
    #     strides=(stride_dvs, stride_dvd),
    #     offsets=(0, 0),
    #     block_shape=(K_TILE_SIZE, DIM),
    #     order=(1, 0)
    # )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    l = tl.load(L_block_ptr, boundary_check=(0, 1), padding_option='zero')
    d = tl.load(D_block_ptr, boundary_check=(0, 1), padding_option='zero')
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option='zero')
    dq = tl.zeros(q.shape, dtype=tl.float32)
    # dk = tl.zeros_like(k)
    # dv = tl.zeros_like(v)

    for key_tile_index in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        # o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option='zero')

        s = tl.dot(q, k.T) * scale
        if is_causal:
            q_star_idx = query_tile_index * Q_TILE_SIZE
            k_star_idx = key_tile_index

            q_pos = q_star_idx + tl.arange(0, Q_TILE_SIZE)[:, None]   # (Q_TILE_SIZE, 1)
            k_pos = k_star_idx + tl.arange(0, K_TILE_SIZE)[None, :]   # (1,   K_TILE_SIZE)

            causal_mask = q_pos >= k_pos
            s = tl.where(causal_mask, s, float('-inf'))
            

        p = tl.exp(s - l)
        # dv += tl.dot(p.T, do)
        dp = tl.dot(do, v.T)
        ds = p * (dp - d)
        dq += tl.dot(ds.to(dtype=k.dtype), k) * scale
        # dk += tl.dot(ds.T, q) * scale

        # Update block pointers for next iteration
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(DQ_block_ptr, dq.to(Q_block_ptr.type.element_ty), boundary_check=(0, 1))
    # tl.store(DK_block_ptr, dk.to(K_block_ptr.type.element_ty), boundary_check=(0, 1))
    # tl.store(DV_block_ptr, dv.to(V_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_kernel_d(
    output, grad_output, D,
    stride_ob, stride_os, stride_od,
    stride_dob, stride_dos, stride_dod,
    stride_db, stride_ds, stride_dd,
    TILE_SIZE: tl.constexpr,
    N_QUERIES: tl.constexpr,
    O_DIM: tl.constexpr
):
    tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        base=output + batch_index * stride_ob,
        shape=(N_QUERIES, O_DIM),
        strides=(stride_os, stride_od),
        offsets=(tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, O_DIM),
        order=(0, 1)
    )

    DO_block_ptr = tl.make_block_ptr(
        base=grad_output + batch_index * stride_dob,
        shape=(N_QUERIES, O_DIM),
        strides=(stride_dos, stride_dod),
        offsets=(tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, O_DIM),
        order=(0, 1)
    )

    D_block_ptr = tl.make_block_ptr(
        base=D + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_ds, stride_dd),
        offsets=(tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, 1),
        order=(0, 1)
    )

    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option='zero')
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option='zero')

    d = tl.sum(o * do, axis=-1, keep_dims=True)

    tl.store(D_block_ptr, d.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        BS, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[-2]

        Q_TILE_SIZE = min(16, N_QUERIES)
        K_TILE_SIZE = min(16, N_KEYS)

        T_q = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        O = torch.empty_like(Q, device=Q.device, dtype=Q.dtype)
        L = torch.empty((BS, N_QUERIES), device=Q.device, dtype=Q.dtype)

        scale = D ** -0.5
        
        ctx.D = D
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.is_causal = is_causal

        flash_fwd_kernel[(T_q, BS)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2), 
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            ctx.D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            ctx.is_causal
        )
        ctx.save_for_backward(L.squeeze(), Q, K, V, O)
        return O

    def backward(ctx, grad_output: torch.Tensor):
        L, Q, K, V, output = ctx.saved_tensors
        L = L[..., None]
        
        BS, N_QUERIES, DIM = Q.shape
        N_KEYS = K.shape[-2]
        O_DIM = grad_output.shape[-1]

        scale = DIM ** -0.5

        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        is_causal = ctx.is_causal

        T_q = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        T_k = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE

        DQ = torch.zeros_like(Q)
        DK = torch.zeros_like(K)
        DV = torch.zeros_like(V)

        D = torch.zeros((*output.shape[:-1], 1), dtype=output.dtype, device=output.device)
        # D = (output * grad_output).sum(dim=-1, keepdim=True) # BUG: need a matrix multiply kernel

        flash_bwd_kernel_d[T_q, BS](
            output, grad_output, D,
            output.stride(0), output.stride(1), output.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            Q_TILE_SIZE,
            N_QUERIES,
            O_DIM
        )

        flash_bwd_kernel_kv[T_k, BS](
            Q, K, V, 
            output, grad_output, L, D,
            DQ, DK, DV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            DQ.stride(0), DQ.stride(1), DQ.stride(2),
            DK.stride(0), DK.stride(1), DK.stride(2),
            DV.stride(0), DV.stride(1), DV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            DIM,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        flash_bwd_kernel_q[T_q, BS](
            Q, K, V, 
            output, grad_output, L, D,
            DQ, DK, DV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            DQ.stride(0), DQ.stride(1), DQ.stride(2),
            DK.stride(0), DK.stride(1), DK.stride(2),
            DV.stride(0), DV.stride(1), DV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            DIM,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        return DQ, DK, DV, None


@torch.compile()
class FlashAttentionPytorch(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        # TODO: write 
        device = Q.device 
        dtype = Q.dtype
        ctx.is_causal = is_causal

        bs, query_len, d_model = Q.shape
        key_len = K.shape[-2]
        scale = d_model ** -0.5

        Q_TILE_SIZE = min(64, query_len)
        K_TILE_SIZE = min(64, key_len)

        T_q = (query_len + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        T_k = (key_len + K_TILE_SIZE - 1) // K_TILE_SIZE

        # Init the output.
        O = torch.empty_like(Q)
        L = torch.zeros((bs, query_len, 1), device=device, dtype=dtype)

        for i in range(T_q):
            # Find the boundary.
            q_start_idx = i * Q_TILE_SIZE
            q_end_idx = min((i + 1) * Q_TILE_SIZE, query_len)
            q = Q[..., q_start_idx:q_end_idx, :]
            
            max_score = torch.full((bs, q_end_idx - q_start_idx, 1), fill_value=-torch.inf, device=device, dtype=dtype)
            sum_exp = torch.zeros((bs, q_end_idx - q_start_idx, 1), device=device, dtype=dtype)
            tile_output = torch.zeros((bs, q_end_idx - q_start_idx, d_model), device=device, dtype=dtype)

            for j in range(T_k):
                # Load k, v tile block.
                k_start_idx = j * K_TILE_SIZE
                k_end_idx = min((j + 1) * K_TILE_SIZE, key_len)
                k = K[..., k_start_idx:k_end_idx, :]
                v = V[..., k_start_idx:k_end_idx, :]

                # Compute attention scores
                attention_score = q @ k.transpose(-1, -2) * scale

                # Use mask if necessary.
                if is_causal:
                    q_pos = torch.arange(q_start_idx, q_end_idx, device=attention_score.device)[:, None]
                    k_pos = torch.arange(k_start_idx, k_end_idx, device=attention_score.device)[None, :]
                    causal_mask =  q_pos >= k_pos
                    attention_score = torch.where(causal_mask, attention_score, float("-inf"))

                # Find the max number in the block
                current_max_score = torch.maximum(max_score, attention_score.max(dim=-1, keepdim=True).values)
                exp_score_stable = torch.exp(attention_score - current_max_score)
                
                # Online softmax, update previous data based on current max_score.
                sum_exp = sum_exp * (torch.exp(max_score - current_max_score)) + exp_score_stable.sum(dim=-1, keepdim=True)
                tile_output = tile_output * (torch.exp(max_score - current_max_score)) + exp_score_stable @ v

                max_score = current_max_score

            O[..., q_start_idx:q_end_idx, :] = tile_output / sum_exp
            L[..., q_start_idx:q_end_idx, :] = torch.log(sum_exp) + max_score
        ctx.save_for_backward(L.squeeze(), Q, K, V, O)
        return O

    @staticmethod
    def backward_no_tile(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        L = L[..., None]
        d = Q.shape[-1]
        scale = d ** -0.5

        S = Q @ K.transpose(-1, -2) * scale         # S:  [query, key]
        if ctx.is_causal:
            mask = torch.triu(torch.ones((Q.shape[-2], K.shape[-2]), device=Q.device), diagonal=1).bool()
            torch.masked_fill(S, mask, value=float('-inf'))

        P = torch.exp(S - L)                        # P:  [query, key]
        dV = P.transpose(-1, -2) @ dO               # dV: [key, d]
        dP = dO @ V.transpose(-1, -2)               # dP: [query, key]
        D = (O * dO).sum(dim=-1, keepdim=True)      # D:  [query, 1]
        dS = P * (dP - D)                           # dS: [query, key]
        dQ = dS @ K * scale                         # dQ: [query, key]
        dK = dS.transpose(-1, -2) @ Q * scale       # dK: [key, d]

        return dQ, dK, dV, None

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        L = L[..., None]
        D = (O * dO).sum(dim=-1, keepdim=True)

        bs, query_len, d_model = Q.shape
        key_len = K.shape[-2]
        scale = d_model ** -0.5

        Q_TILE_SIZE = min(64, query_len)
        K_TILE_SIZE = min(64, key_len)

        T_q = (query_len + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        T_k = (key_len + K_TILE_SIZE - 1) // K_TILE_SIZE

        dq = torch.zeros_like(Q)
        dk = torch.zeros_like(K)
        dv = torch.zeros_like(V)
    
        for j in range(T_k):
            k_start_idx = j * K_TILE_SIZE
            k_end_idx = min(k_start_idx + K_TILE_SIZE, key_len)

            k_tile = K[..., k_start_idx:k_end_idx, :]
            v_tile = V[..., k_start_idx:k_end_idx, :]

            dv_tile = torch.zeros_like(v_tile)
            dk_tile = torch.zeros_like(k_tile)


            for i in range(T_q):

                q_start_idx = i * Q_TILE_SIZE
                q_end_idx = min(q_start_idx + Q_TILE_SIZE, query_len)
                q_tile = Q[..., q_start_idx:q_end_idx, :]
                # dq_tile = torch.zeros_like(q_tile)
                dq_tile = dq[..., q_start_idx:q_end_idx, :]
                
                l_tile = L[..., q_start_idx:q_end_idx, :]
                d_tile = D[..., q_start_idx:q_end_idx, :]
                do_tile = dO[..., q_start_idx:q_end_idx, :]

                s = q_tile @ k_tile.transpose(-1, -2) * scale

                if ctx.is_causal:
                    q_pos = torch.arange(q_start_idx, q_end_idx, device=Q.device)[:, None]
                    k_pos = torch.arange(k_start_idx, k_end_idx, device=Q.device)[None, :]
                    mask = q_pos >= k_pos
                    s = torch.where(mask, s, float('-inf'))
                
                p = torch.exp(s - l_tile)
                dv_tile += p.transpose(-1, -2) @ do_tile
                dp_tile = do_tile @ v_tile.transpose(-1, -2)

                ds_tile = p * (dp_tile - d_tile)
                dq_tile += ds_tile @ k_tile * scale
                dk_tile += ds_tile.transpose(-1, -2) @ q_tile * scale

            dq[..., q_start_idx:q_end_idx, :] = dq_tile
            dk[..., k_start_idx:k_end_idx, :] = dk_tile
            dv[..., k_start_idx:k_end_idx, :] = dv_tile
    
        return dq, dk, dv, None
                

if __name__ == '__main__':
    x = torch.arange(2*4).view(1, 4, 2) / 100
    x = torch.randn((2, 17, 16))
    a_1 = FlashAttentionPytorch.forward(None, x, x, x)
