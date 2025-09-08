import torch
import triton
import triton.language as tl
from triton import cdiv
from einops import rearrange


def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D] 
    return (weight * x).sum(axis=-1)


@triton.jit 
def weighted_sum_fwd(
    x_ptr, weight_ptr, # Input pointers 
    output_ptr, # Output pointer 
    x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor 
    weight_stride_dim, # Likely 1 
    output_stride_row, # Likely 1 
    ROWS, D, 
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Tile shapes must be known at compile time
):

    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in 
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory 
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor 
    # - The overall shape of the tensor to handle out-of-bounds access

    # - The strides of each dimension to use the memory layout properly # - The ND coordinates of the starting block, i.e., "offsets" # - The block shape to use load/store at a time # - The order of the dimensions in memory from major to minor # axes (= np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(ROWS, D,), 
        strides=(x_stride_row, x_stride_dim), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), 
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, 
        shape=(D,), 
        strides=(weight_stride_dim,), 
        offsets=(0,), 
        block_shape=(D_TILE_SIZE,), 
        order=(0,), 
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr, 
        shape=(ROWS,), 
        strides=(output_stride_row,), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE,), 
        block_shape=(ROWS_TILE_SIZE,), 
        order=(0,), 
    )

    # Initialize a buffer to write to 
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer 
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D, 
        # we need boundary checks for both dimensions 
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE) 
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas 
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # Move by D_TILE_SIZE in the last dimension 
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # Move by D_TILE_SIZE
    
    # Write output to the output block pointer (a single scalar per row). 
    # # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks 
    tl.store(output_block_ptr, output, boundary_check=(0,))



@triton.jit 
def weighted_sum_backward(
    x_ptr, weight_ptr, # Input 
    grad_output_ptr, # Grad input 
    grad_x_ptr, partial_grad_weight_ptr, # Grad outputs
    stride_xr, stride_xd, 
    stride_wd, 
    stride_gr, 
    stride_gxr, stride_gxd, 
    stride_gwb, stride_gwd, 
    NUM_ROWS, D, 
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0) 
    n_row_tiles = tl.num_programs(0)

    # Inputs 
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr, 
        shape=(NUM_ROWS,), 
        strides=(stride_gr,), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE,), 
        block_shape=(ROWS_TILE_SIZE,), 
        order=(0,), 
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr, 
        shape=(NUM_ROWS, D,), 
        strides=(stride_xr, stride_xd), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), 
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, 
        shape=(D,), 
        strides=(stride_wd,), 
        offsets=(0,), 
        block_shape=(D_TILE_SIZE,), 
        order=(0,), 
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr, 
        shape=(NUM_ROWS, D,), 
        strides=(stride_gxr, stride_gxd), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), 
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr, 
        shape=(n_row_tiles, D,), 
        strides=(stride_gwb, stride_gwd), 
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )


    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE,)

        # Outer product for grad_x 
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))  # Never out of bounds for dim 0

        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we 
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.

        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D 
        input_shape = x.shape 
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch" 
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors" 
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"


        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension 
        ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time 
        ctx.input_shape = input_shape

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0! 
        y = torch.empty(output_dims, device=x.device)
        
        # Launch our kernel with n instances in our 1D grid. 
        n_rows = y.numel() 
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, 
            y, 
            x.stride(0), x.stride(1), 
            weight.stride(0), 
            y.stride(0), 
            ROWS=n_rows, D=D, 
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = (ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE,)  # These don't have to be the same
        n_rows, D = x.shape

        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight

if __name__ == "__main__":
    x = torch.rand(32, 64).to("cuda")
    w = torch.rand(64).to("cuda")
    print(x)
    print(w)
    y = WeightedSumFunc.apply(x, w)
    print(f"y {y!r}")