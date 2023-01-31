# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd.function import once_differentiable

from adet import _C


class _MSDeformAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                im2col_step):
        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            _C.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights,
                                pred_attentions=None):
    """
    value is sampled and attention is calculated
    :param value: (bs, H1W1+H2W2+..., n_heads, C//n_heads)->
                value_list (n_levles, bs, H_l*W_l, n_heads, C//n_heads) ->
                value_l    (bs*n_heads, C//n_heads, H_l, W_l) ->
                sampling_value_l_  (bs*n_heads, C//n_heads, Nq, n_points) -> torch.stack(dim=-2)
                (bs*n_heads, C//n_heads, Nq, n_levels, n_points)  ->
                (bs*n_heads, C//n_heads, Nq, n_levels*n_points)
    :param value_spatial_shapes:
    :param sampling_locations:
    :param attention_weights: (bs, Nq, n_heads, n_levels, n_points) -> Softmax ->
                              (bs, Nq, n_heads, n_levels, n_points) ->
                              (bs*n_heads, 1, Nq, n_levels*n_points)
    :param pred_attentions: (bs, H1W1+H2W2+..., 1) ->
                            [(bs, H1W1, 1), ...] -> (bs, 1, H1, W1), ... -> (bs*n_heads, 1, H1, W1), ...
                            (bs*n_heads, 1, Nq, n_points), ... -> torch.stack(dim=-2) ->
                            (bs*n_heads, 1, Nq, n_levels, n_points) ->
                            (bs*n_heads, 1, Nq, n_levels*n_points)
                            then add into attention and softmax.

    :return:
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape  # it's also the shape of
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    if pred_attentions is not None:
        # print(f'pred_attentions is Not None!!!')
        pred_list = pred_attentions.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                                          dim=1)  # [(bs, H1W1, 1), ...]
        sampling_pred_list = []
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
        if pred_attentions is not None:
            pred_l = pred_list[lid_].transpose(1, 2).reshape(N_, 1, H_, W_).repeat(M_, 1, 1, 1)
            sampling_pred_l = F.grid_sample(pred_l, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_pred_list.append(sampling_pred_l)

    attention_weights = attention_weights.transpose(1, 2).flatten(0, 1).reshape(N_ * M_, 1, Lq_, L_, P_)
    if pred_attentions is not None:
        pred_to_attn_weights = torch.stack(sampling_pred_list, dim=-2)  # (bs*n_heads, 1, Nq, n_levels, n_points)
        attention_weights = attention_weights + pred_to_attn_weights
    attention_weights = attention_weights.flatten(-2)
    attention_weights = F.softmax(attention_weights, -1)   # (N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_,
                                                                                                     Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, mode='cuda'):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        self.mode = mode
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                              self.n_levels,
                                                                                                              self.n_points,
                                                                                                              1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None, pred_attentions=None):
        """
        :param pred_attentions (tesor):    (N, length_{query}, 1)
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1],
                                            top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4),
                                            add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ),
                       [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l),
                                            True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape  # 有位置编码信息 -> query -> attention
        N, Len_in, _ = input_flatten.shape  # 无位置编码信息 -> value
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # 全局预测影响offsets：效果不好
        # sampling_offsets = self.sampling_offsets(
        #     query + pred_attentions if pred_attentions is not None else query).view(N, Len_q, self.n_heads,
        #                                                                             self.n_levels, self.n_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads,
                                                             self.n_levels, self.n_points, 2)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
<<<<<<< HEAD
        # output = _MSDeformAttnFunction.apply(
        #     value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = ms_deform_attn_core_pytorch(value=value,
                                             value_spatial_shapes=input_spatial_shapes,
                                             sampling_locations=sampling_locations,
                                             attention_weights=attention_weights)
=======
        if self.mode == 'pytorch':
            attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights, pred_attentions)
        elif self.mode == 'cuda':
            attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads,
                                                                   self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads,
                                                                      self.n_levels, self.n_points)
            output = _MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,
                                                 sampling_locations, attention_weights, self.im2col_step)
        else:
            raise ValueError('mode can only be in ["pytorch", "cuda"]')
>>>>>>> 801a65d4518f9921daaae9e14b2dfdeb6af1ab8b
        output = self.output_proj(output)
        return output


class MSDeformAttn_v2(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4, mode='cuda'):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        self.mode = mode
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1,
                                                                                                          self.n_points,
                                                                                                          1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None, pred_attentions=None, indexes=None):
        """
        :param indexes List(Tensor):     4个level，每个元素 bs*n_head, 4, h*w
        :param pred_attentions (tesor):    (N, length_{query}, 1)
        :param query                       (N, Length_{query}, C)
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ),
                       [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l),
                                            True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape  # 有位置编码信息 -> query -> attention
        N, Len_in, _ = input_flatten.shape  # 无位置编码信息 -> value
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)    # bs, h1w1+..., c
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_points)
        output = ms_deform_attn_core_pytorch_v2(
            value, input_spatial_shapes, attention_weights, pred_attentions, indexes)
        output = self.output_proj(output)
        return output


def ms_deform_attn_core_pytorch_v2(value, value_spatial_shapes, attention_weights,
                                pred_attentions=None, indexes=None):
    """
    value is sampled and attention is calculated
    :param value: (bs, H1W1+H2W2+..., n_heads, C//n_heads)->
                value_list (n_levles, bs, H_l*W_l, n_heads, C//n_heads) ->
                value_l    (bs*n_heads, C//n_heads, H_l, W_l) ->
                sampling_value_l_  (bs*n_heads, C//n_heads, Nq, n_points) -> torch.stack(dim=-2)
                (bs*n_heads, C//n_heads, Nq, n_levels, n_points)  ->
                (bs*n_heads, C//n_heads, Nq, n_levels*n_points)
    :param value_spatial_shapes:
    :param attention_weights: (bs, h1w1+h2w2+..., self.n_heads, self.n_points)
    :param pred_attentions: (bs, H1W1+H2W2+..., 1) ->
                            [(bs, H1W1, 1), ...] -> (bs, 1, H1, W1), ... -> (bs*n_heads, 1, H1, W1), ...
                            (bs*n_heads, 1, Nq, n_points), ... -> torch.stack(dim=-2) ->
                            (bs*n_heads, 1, Nq, n_levels, n_points) ->
                            (bs*n_heads, 1, Nq, n_levels*n_points)
                            then add into attention and softmax.

    :return:
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape  # bs, h1w1+h2w2+..., n_head, c//n_head
    P_ = indexes[0].size()[-2]
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    if pred_attentions is not None:
        sampling_pred_list = []
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # sample value_l_
        value_unfolded = F.unfold(value_l_, 3, padding=1)  # bs*n_head, 9*c//n_head, H_*W_
        sampling_value_l_ = value_unfolded.reshape(N_*M_, D_, 9, H_*W_).gather(2, indexes[lid_])
        # bs*n_head, c//n_head, 4, H_*W_

        sampling_value_list.append(sampling_value_l_.transpose(2, 3))
        if pred_attentions is not None:
            pred_l = pred_attentions[lid_].unsqueeze(1)
            # bs*n_head, 4, h*w -> bs*n_head, 1, 4, h*w
            sampling_pred_list.append(pred_l)

    attention_weights = attention_weights.transpose(1, 2).flatten(0, 1).reshape(N_ * M_, 1, S_, P_)
    if pred_attentions is not None:
        pred_to_attn_weights = torch.cat(sampling_pred_list, dim=-1)  # bs*n_head, 1, 4, h1w1+h2w2+...
        attention_weights = attention_weights + pred_to_attn_weights.permute(0, 1, 3, 2)
    attention_weights = F.softmax(attention_weights, -1)  # (N_ * M_, 1, S_, P_)
    output = (torch.cat(sampling_value_list, dim=-2) * attention_weights).sum(-1).view(N_, M_ * D_, S_)
    # bs*n_head, c//n_head, h1w1+h2w2+..., 4  * bs*n_head, 1, h1w1+h2w2+..., 4
    return output.transpose(1, 2).contiguous()
