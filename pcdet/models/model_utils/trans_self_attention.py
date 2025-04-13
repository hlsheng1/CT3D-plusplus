import torch.nn as nn
import ipdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from copy import deepcopy


class TransSF(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_1, src_2):
        bs, n, c = src_1.shape

        # ipdb.set_trace()
        # channel-wise, we generate attention values for different channels
        src_1 = src_1.permute(1, 0, 2)
        src_2 = src_2.permute(1, 0, 2)

        out = self.encoder(src_1, src_2)
        # ipdb.set_trace()
        return out.permute(1, 0, 2).contiguous()


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src_l, src_i):


        for layer in self.layers:
            output = layer(src_l, src_i)

        if self.norm is not None:
            output = self.norm(output)


        return output



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn_l = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linearl1 = nn.Linear(d_model, dim_feedforward)
        self.linearl2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)

        self.norml1 = nn.LayerNorm(d_model)
        self.norml2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src_l, src_i, 
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = src_l
        k = src_i
        v = src_i
        src_l0 = self.self_attn_l(q, k, value=v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src_l1 = src_l + self.dropout1(src_l0)
        src_l1 = self.norml1(src_l1)
        src_l2 = self.linearl2(self.dropout(self.activation(self.linearl1(src_l1))))
        src_l3 = src_l + self.dropout2(src_l2)

        return src_l3

    def forward_pre(self, 
                    src_l, src_i, 
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        q0 = src_l
        k0 = src_i
        v0 = src_i
        v0 = self.norml1(v0)
        # ipdb.set_trace()

        src_i0 = self.self_attn_l(q0, k0, value=v0, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_l1 = src_l + self.dropout1(src_i0)
        src_l1 = self.norml2(src_l1)
        src_l2 = self.linearl2(self.dropout(self.activation(self.linearl1(src_l1))))
        src_l3 = src_l + self.dropout2(src_l2)

        return src_l3

    def forward(self, src_l, src_i):
        if self.normalize_before:
            return self.forward_pre(src_l, src_i)
        return self.forward_post(src_l, src_i)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_sf():
    return TransSF(
        d_model=256,
        dropout=0.1,
        nhead=4,
        dim_feedforward=512,
        num_encoder_layers=1,
        normalize_before=True,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")