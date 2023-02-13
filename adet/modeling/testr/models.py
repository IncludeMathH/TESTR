import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from adet.layers.deformable_transformer import DeformableTransformer

from adet.layers.pos_encoding import PositionalEncoding1D
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TESTR(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """

    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.use_attention = cfg.MODEL.ATTENTION.ENABLED

        self.backbone = backbone

        # fmt: off
        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes = 1
        self.max_text_len = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset = not cfg.MODEL.TRANSFORMER.USE_POLYGON    # 默认USE_POLYGON=true, 故此值为False

        self.text_pos_embed = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on

        # ========通过特征图得到全局Mask的概率=======
        use_attention_in_transformer = False
        if self.use_attention:
            self.pred_attention = nn.ModuleList([nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=1)
                                                 for _ in range(self.num_feature_levels)])
            use_attention_in_transformer = cfg.MODEL.ATTENTION.IN_TRANSFORMER
        mode = cfg.MODEL.mode
        self.transformer = DeformableTransformer(
            d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals, use_attention=use_attention_in_transformer,
            mode=mode, window_size=cfg.MODEL.ATTENTION.window_size, 
        )            # 暂时不考虑层间采样！！
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)
        self.text_class = nn.Linear(self.d_model, self.voc_size + 1)

        # shared prior between instances (objects)
        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model),
                ))
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)])
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.to(self.device)

        # ===============使用高斯滤波========
        use_gaussian = False
        if self.use_attention:
            use_gaussian = cfg.MODEL.ATTENTION.USE_GAUSSIAN
        if use_gaussian:
            self.conv2d_gaussian = nn.Conv2d(1, 1, (3, 3), padding=1)
            w = 1/4.8976 * torch.Tensor([[[[0.3679, 0.6065, 0.3679],
                                           [0.6065, 1., 0.6065],
                                           [0.3679, 0.6065, 0.3679]]]])
            self.conv2d_gaussian.weight = nn.Parameter(w, requires_grad=False)
        self.use_gaussian = use_gaussian

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_keypoints": The normalized keypoint coordinates for all queries, represented as
                               (x, y). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)        # len(features)=3
        # print(f'features has {len(features)} levels!')

        if self.num_feature_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()     # src and mask have the same shape. (bs, 512, h1, w1), (bs, h1, w1), ...
            # print(f'the shape of src:{src.shape}, the shape of mask:{mask.shape}, the shape of pos[l]:{pos[l].shape}')
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]   # bs, h4, w4
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                # print(f'pos_l = {pos_l.shape}')
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        # if use global attention, then do:
        if self.use_attention:
            pred_attentions = []
            if self.use_gaussian:
                pred_attentions_gaussian = []
            for lvl, src in enumerate(srcs):            # src is supposed to be (bs, c_l, H_l, W_l)
                pred_attention = self.pred_attention[lvl](src)   # (bs, 1, hl, wl)
                pred_attentions.append(pred_attention)
                if self.use_gaussian:
                    pred_attentions_gaussian.append(self.conv2d_gaussian(pred_attention))
        else:
            pred_attentions = None
        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None,
            pred_attentions=pred_attentions_gaussian if self.use_gaussian else pred_attentions)
        # hs: Tensor, the query output of ctrl_point_embed (n_layer, bs, n_proposals, n_ctrl_points, c)
        # hs_text: Tensor the query output of text_embed (n_layer, bs, n_proposals, max_len, c)
        # init_reference: the coarse prediction of encoder, has the shape of (bs, n_proposal, 4) 4: cx, cy, w, h
        # inter_references: List(Tensor), 似乎每个元素都是init_reference (n_layer, bs, n_proposal, 4)
        # enc_outputs_class: the output of encoder pass through o linear layer (bs, h1w1+..., 1)
        # enc_outputs_coord_unact: (bs, h1w1+h2w2+h3w3+h4w4, 4)
        # 测试
        # a = init_reference - inter_references[0]
        # print(f' the unique value of init and inter:{a.unique()}')
        """
        the unique value of init and inter:tensor([0.], device='cuda:0')
        """
        # print(f'the shape of hs_text: {hs_text.shape}')

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)    # 反sigmoid. torch.log(x/(1-x))
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])     # 每个self.ctrl_point_class都是线性层
            tmp = self.ctrl_point_coord[lvl](hs[lvl])  # 每个coord都是MLP -> (bs, n_proposals, n_ctrl_points, 2) 预测的应
            # 该是偏置
            if reference.shape[-1] == 2:    # reference:(bs, n_proposals, 4)
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]      # reference -> (bs, n_proposal, 1, 2)
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset) # modified sigmoid for range [-0.5, 1.5]
            # 此处为：tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_text = torch.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],
               'pred_texts': outputs_text[-1],
               }
        if self.use_attention:
            out['pred_attentions'] = pred_attentions
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]
