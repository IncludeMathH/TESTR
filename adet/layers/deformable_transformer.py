# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.nn.modules.container import T

from adet.utils.misc import inverse_sigmoid
from .ms_deform_attn import MSDeformAttn, MSDeformAttn_v2


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 num_proposals=300, use_attention=False, mode='cuda'):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_proposals = num_proposals

        self.use_attention = use_attention
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, use_attention,
                                                          mode=mode)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableCompositeTransformerDecoderLayer(d_model, dim_feedforward,
                                                                   dropout, activation,
                                                                   num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableCompositeTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.bbox_class_embed = None
        self.bbox_embed = None
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model, d_model)
        self.pos_trans_norm = nn.LayerNorm(d_model)

        self._reset_parameters()
        self.parse_mask = nn.Conv2d(1, self.nhead, (3, 3), padding=1)     # TODO: ReLU
        self.enc_n_points = enc_n_points

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        """

        :param proposals (Tensor): (bs, 100, 4)
        :return:
        """
        num_pos_feats = 64
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """

        :param memory (Tensor): (bs, n, c)
        :param memory_padding_mask (Tensor): （bs, h1w1+h2w2+h3w3+h4w4)
                                            True for padding elements, False for non-padding elements
        :param spatial_shapes List(Tensor):
        :return:
        """
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)      # H方向可用的像素数  (bs, )
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)      # W方向可用的像素数  (bs, )

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)   # (H_, W_, 2)  直观意义的坐标矩阵

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale      # 复制bs次，然后除以横纵可用像素数; 加了0.5
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)      # (bs, H_, W_, 2)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)  # (bs, H_, W_, 4) -> (bs, H_*W_, 4)
                                                                  # 4: (x, y, w, h),归一化
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)   # (bs, h1w1+h2w2+..., 4) <==> (bs, n_q, 4)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # 检验是否都是True
        output_proposals = torch.log(output_proposals / (1 - output_proposals))      # [反sigmoid]
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        # ==============以上都是凭借尺寸和mask信息得出===========

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))      # nn.Linear(d_model, d_model)
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """
        因为每一层的特征涉及到补零的问题，此函数用来计算可用的部分占整个长宽的比例。
        :param mask(tensor): (bs, hl, wl), 取值为bool，True表示补零的部分
        :return: valid_ratio(tensor): (bs, 2) 每一行是：w方向的可用比例，h方向的可用比例
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, text_embed, text_pos_embed, text_mask=None, pred_attentions=None):
        """

        :param srcs List(Tensor): Tensor: (b, c, hl, wl)
        :param masks List(Tensor): 同样从backbone得到，标记哪些位置是补零的结果而哪些不是
        :param pos_embeds List(Tensor): 从backbone得到，属于黑盒子
        :param query_embed:(Tensor)  ctrl_point_embed  (self.num_proposals, self.num_ctrl_points, self.d_model)
                    nn.Embedding(self.num_ctrl_points, self.d_model).weight[None, ...].repeat(self.num_proposals, 1, 1)
        :param text_embed:(Tensor) similar to query_ebed (self.num_proposals, self.max_text_len, self.d_model)
        :param text_pos_embed:(Tensor) 对text_embed进行位置编码 (self.num_proposals, self.max_text_len, self.d_model)
        :param text_mask:
        :param pred_attentions:
        :return:
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        pred_attentions_flatten = []
        indexes = []
        for lvl, (src, mask, pos_embed, pred_attention) in enumerate(zip(srcs, masks, pos_embeds, pred_attentions)):
            bs, c, h, w = src.shape  # src 是特征图
            spatial_shape = (h, w)  # 记录特征图的
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # bs, c, h, w -> bs, h*w,  c
            mask = mask.flatten(1)  # 还不知道为何有mask
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, c, h, w-> bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            # ==============prepare index gathered from global mask ================
            pred_attention = self.parse_mask(pred_attention)         # bs, n_head, h, w
            mask_folded = F.unfold(pred_attention, 3, padding=1).reshape(bs, self.nhead, 9, h * w).flatten(0, 1)
            # bs, 9*n_head, 35
            _, index = mask_folded.topk(self.enc_n_points, dim=1)  # bs*n_head, 4, h*w
            indexes.append(index.unsqueeze(1).repeat(1, self.d_model//self.nhead, 1, 1))
            # ==================use global attention and sampled ===============================
            pred_attention_folded = F.unfold(pred_attention.flatten(0, 1).unsqueeze(1), 3, padding=1)
            # bs, n_head, h, w -> bs*n_head, 1, h, w -> bs*n_head, 9, h*w
            sampled_pred = pred_attention_folded.gather(1, index)  # bs*n_head, 4, h*w
            pred_attentions_flatten.append(sampled_pred)  # # bs*n_head, 4, h*w
        src_flatten = torch.cat(src_flatten, 1)  # bs, h1w1+h2w2+h3w3+h4w4, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, h1w1+h2w2+h3w3+h4w4
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # 从哪里开始是哪个level，可以用于复原
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)     # (bs, num_levels, 2) 当b=1时，都是1，无用

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten, pred_attentions_flatten, indexes)

        # prepare input for decoder
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # (bs, n_q, c), (bs, n_q, 4)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.bbox_class_embed(output_memory)  # nn.Linear(self.d_model, self.num_classes)
        # num_classes = 1  实际上获得的是概率
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals  # (bs, n_q, 4), 前者反而是主要的
        # bbox_embed: MLP(self.d_model, self.d_model, 4, 3)  (input_dim, hidden_dim, output_dim, num_layers)

        topk = self.num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        # (bs, n_q) 返回索引 -> (bs, 100)
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        # (bs, 100, 4)
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points
        query_pos = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        # get_proposal_pos_embed: (bs, 100, 4) -> (bs, 100, 256)
        # pos_trans (nn.Linear(256, 256)): (bs, 100, 256) -> (bs, 100, 256)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1, -1)
        # (100, self.num_ctrl_points, 256) -> (bs, 100, self.num_ctrl_points, 256)
        query_pos = query_pos[:, :, None, :].repeat(1, 1, query_embed.shape[2], 1)
        text_embed = text_embed.unsqueeze(0).expand(bs, -1, -1, -1)
        # (100, self.max_len, 256) -> (bs, 100, self.max_len, 256)

        # decoder
        hs, hs_text, inter_references = self.decoder(
            query_embed, text_embed, reference_points, memory, spatial_shapes,
            level_start_index, valid_ratios, query_pos, text_pos_embed, mask_flatten, text_mask
        )   # output, output_text, reference_points

        inter_references_out = inter_references
        return hs, hs_text, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, use_attention=False, mode='cuda', index=0):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn_v2(d_model, n_heads, n_points, mode)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # whether to use global attention
        self.use_attention = use_attention

        # 表明该layer是第几层
        self.index = index
        self.n_levels = n_levels

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, spatial_shapes, level_start_index, padding_mask=None, pred_attentions=None, indexes=None):
        """

        :param src: (bs, h1w1+h2w2+h3w3,c)
        :param pos: (bs, h1w1+h2w2+h3w3,c)  位置编码
        :param reference_points: 只是[-1, 1)范围内的格子罢了
        :param spatial_shapes:
        :param level_start_index:
        :param padding_mask:
        :param pred_attentions:
        :return:
        """
        # self attention
        # ========loss+线性层作用于Q=========
        # if self.use_attention:
        #     assert pred_attentions is not None
        #     src_attention = src * self.attn_map(pred_attentions)
        # src2 = self.self_attn(self.with_pos_embed(src_attention, pos), reference_points, src, spatial_shapes, level_start_index,
        #                       padding_mask)
        src_value = src
        if self.use_attention:
            assert pred_attentions is not None
            # src_value = src * self.attn_map(pred_attentions)
        src2 = self.self_attn(self.with_pos_embed(src, pos), src_value, spatial_shapes, level_start_index,
                              padding_mask, pred_attentions, indexes)
        # feature fusion
        src2_list = src2.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        bs, n_q, c = src.shape
        src2_tmp = []
        for lid_, (H_, W_) in enumerate(spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            src2_list_ = src2_list[lid_].transpose(1, 2).reshape(bs, c, H_, W_)
            src2_tmp.append(src2_list_)
        if self.index % 2 == 0:
            for i in range(self.n_levels-1):        # 0, 1, 2
                src2_tmp[i+1] = src2_tmp[i+1] + F.interpolate(src2_tmp[i], size=src2_tmp[i+1].shape[-2:], mode='bilinear')
        else:
            for i in range(self.n_levels-1, 0, -1):    # 3, 2, 1
                src2_tmp[i-1] = src2_tmp[i-1] + F.interpolate(src2_tmp[i], size=src2_tmp[i-1].shape[-2:], mode='bilinear')
        src2_flatten = []
        for src2_tmp_ in src2_tmp:
            src2_flatten.append(src2_tmp_.flatten(2).transpose(1, 2))    # bs, h*w, c
        src2 = torch.cat(src2_flatten, dim=1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # 标明各layer的层数
        for i in range(self.num_layers):
            self.layers[i].index = i

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                pred_attentions=None, indexes=None):
        """
            (src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                          mask_flatten, pred_attentions_flatten, indexes)
        :param src: (bs, h1w1+h2w2+h3w3, c); src_flatten
        :param spatial_shapes: List(Tensor) every element is (hl, wl).
        :param level_start_index:
        :param valid_ratios:
        :param pos: 位置编码+层次编码 (bs, h1w1+h2w2+h3w3, c); lvl_pos_embed_flatten
        :param padding_mask: 推测是哪些位置是补零得到的 (bs, h1w1+h2w2+h3w3); mask_flatten
        :param pred_attentions:
        :return:
        """
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, spatial_shapes, level_start_index, padding_mask, pred_attentions, indexes)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DeformableCompositeTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, mode='cuda'):
        super().__init__()

        ## attn for location branch
        # cross attention
        self.attn_cross = MSDeformAttn(d_model, n_levels, n_heads, n_points, mode)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(d_model)

        # self attention (intra)
        self.attn_intra = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_intra = nn.Dropout(dropout)
        self.norm_intra = nn.LayerNorm(d_model)

        # self attention (inter)
        self.attn_inter = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        ## (factorized) attn for text branch
        ## TODO: different embedding dim for text/loc?
        # attention between text embeddings belonging to the same object query
        self.attn_intra_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_intra_text = nn.Dropout(dropout)
        self.norm_intra_text = nn.LayerNorm(d_model)

        # attention between text embeddings on the same spatial position of different objects
        self.attn_inter_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_inter_text = nn.Dropout(dropout)
        self.norm_inter_text = nn.LayerNorm(d_model)

        # cross attention for text
        self.attn_cross_text = MSDeformAttn(d_model, n_levels, n_heads, n_points, mode)
        self.dropout_cross_text = nn.Dropout(dropout)
        self.norm_cross_text = nn.LayerNorm(d_model)

        # ffn
        self.linear1_text = nn.Linear(d_model, d_ffn)
        self.activation_text = _get_activation_fn(activation)
        self.dropout3_text = nn.Dropout(dropout)
        self.linear2_text = nn.Linear(d_ffn, d_model)
        self.dropout4_text = nn.Dropout(dropout)
        self.norm3_text = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ffn_text(self, tgt):
        tgt2 = self.linear2_text(self.dropout3_text(self.activation_text(self.linear1_text(tgt))))
        tgt = tgt + self.dropout4_text(tgt2)
        tgt = self.norm3_text(tgt)
        return tgt

    def forward(self, tgt, query_pos, tgt_text, query_pos_text, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None, text_padding_mask=None):
        """

        :param tgt: batch_size, n_q, n_control_points, embed_dim
        :param query_pos: batch_size, n_q, n_points, embed_dim
        :param tgt_text: batch_size, n_q, n_words, embed_dim
        :param query_pos_text: batch_size, n_objects, n_words, embed_dim
        :param reference_points: (bs, n_q, n_levels, 4)
        :param src:
        :param src_spatial_shapes:
        :param level_start_index:
        :param src_padding_mask:
        :param text_padding_mask: batch_size, n_objects, n_words
        :return:
        """

        # self attention (intra)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.attn_intra(
            q.flatten(0, 1).transpose(0, 1),
            k.flatten(0, 1).transpose(0, 1),
            tgt.flatten(0, 1).transpose(0, 1),
        )[0].transpose(0, 1).reshape(q.shape)
        tgt = tgt + self.dropout_intra(tgt2)   # bs, n_q, n_control_points, c
        tgt = self.norm_intra(tgt)

        q_inter = k_inter = tgt_inter = torch.swapdims(tgt, 1, 2)  # bs, n_control_points, n_q, c
        tgt2_inter = self.attn_inter(
            q_inter.flatten(0, 1).transpose(0, 1),
            k_inter.flatten(0, 1).transpose(0, 1),
            tgt_inter.flatten(0, 1).transpose(0, 1),
        )[0].transpose(0, 1).reshape(q_inter.shape)
        tgt_inter = tgt_inter + self.dropout_inter(tgt2_inter)
        tgt_inter = torch.swapdims(self.norm_inter(tgt_inter), 1, 2) # bs, n_q, n_control_points, c

        # cross attention
        reference_points_loc = reference_points[:, :, None, :, :].repeat(1, 1, tgt_inter.shape[2], 1, 1)
        tgt2 = self.attn_cross(self.with_pos_embed(tgt_inter, query_pos).flatten(1, 2),
                               reference_points_loc.flatten(1, 2),
                               src, src_spatial_shapes, level_start_index, src_padding_mask).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout_cross(tgt2)
        tgt = self.norm_cross(tgt_inter)

        # text branch - intra self attn (word-wise)
        q_text = k_text = self.with_pos_embed(tgt_text, query_pos_text)
        tgt2_text = self.attn_intra_text(
            q_text.flatten(0, 1).transpose(0, 1),
            k_text.flatten(0, 1).transpose(0, 1),
            tgt_text.flatten(0, 1).transpose(0, 1),
            text_padding_mask.flatten(0, 1) if text_padding_mask is not None else None,
        )[0].transpose(0, 1).reshape(tgt_text.shape)
        tgt_text = tgt_text + self.dropout_intra_text(tgt2_text)
        tgt_text = self.norm_intra_text(tgt_text)

        # text branch - inter self attn (object-wise)
        q_text_inter = k_text_inter = tgt_text_inter = torch.swapdims(tgt_text, 1, 2)
        tgt2_text_inter = self.attn_inter_text(
            q_text_inter.flatten(0, 1).transpose(0, 1),
            k_text_inter.flatten(0, 1).transpose(0, 1),
            tgt_text_inter.flatten(0, 1).transpose(0, 1),
            torch.swapdims(text_padding_mask, 1, 2).flatten(0, 1) if text_padding_mask is not None else None,
        )[0].transpose(0, 1).reshape(q_text_inter.shape)
        tgt_text_inter = tgt_text_inter + self.dropout_inter_text(tgt2_text_inter)
        tgt_text_inter = torch.swapdims(self.norm_inter_text(tgt_text_inter), 1, 2)

        # text branch - cross attn
        reference_points_text = reference_points[:, :, None, :, :].repeat(1, 1, tgt_text_inter.shape[2], 1, 1)
        tgt2_text_cm = self.attn_cross_text(self.with_pos_embed(tgt_text_inter, query_pos_text).flatten(1, 2),
                                            reference_points_text.flatten(1, 2),
                                            src, src_spatial_shapes, level_start_index, src_padding_mask).reshape(
            tgt_text_inter.shape)

        tgt_text_inter = tgt_text_inter + self.dropout_cross_text(tgt2_text_cm)
        tgt_text = self.norm_cross_text(tgt_text_inter)

        # ffn
        tgt = self.forward_ffn(tgt)
        tgt_text = self.forward_ffn_text(tgt_text)

        return tgt, tgt_text


class DeformableCompositeTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        """

        :param decoder_layer:
        :param num_layers:
        :param return_intermediate: 默认是True
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, tgt_text, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, query_pos_text=None, src_padding_mask=None, text_padding_mask=None):
        """
            query_embed, text_embed, reference_points, memory, spatial_shapes,
            level_start_index, valid_ratios, query_pos, text_pos_embed, mask_flatten, text_mask
        )
        :param tgt: query_embed / ctrl_point_embed; (bs, n_proposals, n_ctrl_points, c)
        :param tgt_text: text_embed; (bs, n_proposals, n_max_len, c)
        :param reference_points: (bs, n_proposals, 4)  4: x, y, w, h
        :param src: memory; (bs, n_q, c)
        :param src_spatial_shapes: List(Tensor)
        :param src_level_start_index:
        :param src_valid_ratios: (bs, n_levels, 2)
        :param query_pos: (bs, n_proposals, n_ctrl_points, c)
        :param query_pos_text: text_pos_embed; (n_proposals, max_len, c)
        :param src_padding_mask: (bs, n_q)
        :param text_padding_mask: None
        :return:
        """
        output, output_text = tgt, tgt_text      # query_embed, text_embed

        intermediate = []
        intermediate_text = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                # (bs, n_q, n_levels, 4)
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output, output_text = layer(output, query_pos, output_text, query_pos_text, reference_points_input, src,
                                        src_spatial_shapes, src_level_start_index, src_padding_mask, text_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_text.append(output_text)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_text), torch.stack(intermediate_reference_points)

        return output, output_text, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
