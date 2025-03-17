# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Hyunseop Kim
# ------------------------------------------------------------------------

from typing import Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from mmcv.cnn import build_norm_layer, xavier_init
from .cross_attention import *


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        

class MLP(nn.Module):
    def __init__(self, in_model, dim_forward, out_model, norm=False):
        super().__init__()
        self.in_model = in_model
        self.dim_forward = dim_forward
        self.out_model = out_model

        self.lin_1 = nn.Linear(self.in_model, self.dim_forward)
        self.lin_2 = nn.Linear(self.dim_forward, self.out_model)

        self.relu = F.relu
        if norm:
            self.norm = nn.LayerNorm(self.dim_forward)
        else:
            self.norm = None
        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x):
        x = self.lin_1(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        x = self.lin_2(x)
        return x


class DFFL(nn.Module):
    def __init__(self, dim_in, dim_ctx, dim_out=256):
        super(DFFL, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self.reduce = nn.Sequential(
            nn.Linear(dim_ctx, dim_out),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(dim_out, dim_out)
        self.beta = nn.Linear(dim_out, dim_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self._layer(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta
        return out

class Decoder(nn.Module):
    def __init__(self, in_dims=256, out_dims=256, dim_feedforward=512, num_layer=6, nhead=8, norm=None):
        super().__init__()

        layer = DecoderLayer(in_dims, out_dims, dim_feedforward, nhead)
        self.layers = _get_clones(layer, num_layer)
        self.norm = norm    # None

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, tgt, memory, attn_mask=None):
        intermediate = []
        attention_scores = []
        for layer in self.layers:
            tgt, attn = layer(tgt, memory, attn_mask)
            intermediate.append(tgt)
            attention_scores.append(attn)
        return torch.stack(intermediate), torch.stack(attention_scores)

class DecoderLayer(nn.Module):
    def __init__(self, in_model=256, out_model=256, dim_feedforward=512, nhead=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(in_model, out_model, nhead)
        self.linear1 = nn.Linear(in_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_model)

        self.norm1 = nn.LayerNorm(in_model)
        self.norm2 = nn.LayerNorm(in_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, attn_mask=None):
        tgt2, attention = self.cross_attn(
            q=tgt,
            k=memory,
            v=memory,
            mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, attention


class Encoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, dim_feedforward=512, num_layer=6, nhead=8, norm=None):
        super().__init__()

        layer = EncoderLayer(in_dim, out_dim, dim_feedforward, nhead)
        self.layers = _get_clones(layer, num_layer)
        self.post_norm = norm

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, query, attn_mask=None):
        for layer in self.layers:
            query, w = layer(query, attn_mask)
        return query, w

class EncoderLayer(nn.Module):
    def __init__(self, in_model=256, out_model=256, dim_feedforward=512, nhead=8, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(in_model, out_model, nhead)

        self.linear1 = nn.Linear(in_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_model)

        self.norm1 = nn.LayerNorm(in_model)
        self.norm2 = nn.LayerNorm(in_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, attn_mask=None):
        src2, w = self.self_attn(
            q=src,
            k=src,
            v=src,
            mask=attn_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, w


class FF_module(nn.Module):
    def __init__(self, in_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_model)

        self.norm1 = nn.LayerNorm(in_model)
        self.norm2 = nn.LayerNorm(in_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src2):
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class FFN(nn.Module):
    def __init__(self, in_model=256, dim_feed_forward=512, add_identity=True):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_model, out_features=dim_feed_forward, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False)
        )
        self.layer = nn.Linear(in_features=dim_feed_forward, out_features=in_model, bias=True)
        self.drop_out = nn.Dropout(0.1, inplace=False)
        self.dropout_layer = nn.Identity() #nn.Dropout(0.1, inplace=False)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        x = self.activation(x)
        x = self.layers(x)
        x = self.layer(x)
        x = self.drop_out(x)
        if not self.add_identity:
            return self.dropout_layer(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])