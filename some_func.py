import torch
import math


def get_proposal_pos_embed(proposals):
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


proposals = torch.randn((2, 100, 4))
pos = get_proposal_pos_embed(proposals)
# print(f'proposals = {proposals}')
print(f'pos = {pos.shape}')


# 位置编码
