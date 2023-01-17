import torch
from torch.autograd.function import once_differentiable

from adet import _C

class _MSDeformAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            _C.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


batch_size = 1
len_q = 100
len_in = 100       # len_in 要等与shape之和
n_heads = 4
d_model = 64

device_id = 0

value = torch.rand(batch_size, len_in, n_heads, d_model//n_heads).to(device=device_id)
input_spatial_shapes = torch.tensor([[10, 10]]*batch_size, device=device_id)
attention_weights = torch.rand(batch_size, len_q, n_heads, 4, 4).to(device=device_id)
sampling_locations = torch.rand(batch_size, len_q, n_heads, 4, 4, 2).to(device=device_id)
input_level_start_index = torch.tensor([0, 100], device=device_id)
im2col_step = 64
ouput = _MSDeformAttnFunction.apply(
    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, im2col_step
)
print(ouput)
