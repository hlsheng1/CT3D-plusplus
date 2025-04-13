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


class CWCTransformer(nn.Module):

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
        src_1 = src_1.permute(1, 0, 2)
        src_2 = src_2.permute(1, 0, 2)

        out_1, out_2 = self.encoder(src_1, src_2)
        
        return out_1.permute(1, 2, 0).view(bs, c, n), out_2.permute(1, 2, 0).view(bs, c, n)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src_i, src_l):
        output_i = src_i
        output_l = src_l

        for layer in self.layers:
            output_i, output_l = layer(output_i, output_l)

        if self.norm is not None:
            output_i = self.norm(output_i)
            output_l = self.norm(output_l)

        return output_i, output_l



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn_i = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_l = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.lineari1 = nn.Linear(d_model, dim_feedforward)
        self.lineari2 = nn.Linear(dim_feedforward, d_model)
        self.linearl1 = nn.Linear(d_model, dim_feedforward)
        self.linearl2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)

        self.normi1 = nn.LayerNorm(d_model)
        self.normi2 = nn.LayerNorm(d_model)
        self.norml1 = nn.LayerNorm(d_model)
        self.norml2 = nn.LayerNorm(d_model)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src_i, src_l, 
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q0 = src_l
        k0 = src_i
        v0 = src_i
        src_i0 = self.self_attn_i(q0, k0, value=v0, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        q1 = src_i
        k1 = src_l
        v1 = src_l
        src_l0 = self.self_attn_l(q1, k1, value=v1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src_l1 = src_l + self.dropout1(src_i0)
        src_l1 = self.norml1(src_l1)
        src_l2 = self.linearl2(self.dropout(self.activation(self.linearl1(src_l1))))
        src_l3 = src_l1 + self.dropout2(src_l2)
        src_l3 = self.norml2(src_l3)

        src_i1 = src_i0 + self.dropout1(src_l0)
        src_i1 = self.normi1(src_i1)
        src_i2 = self.lineari2(self.dropout(self.activation(self.lineari1(src_i1))))
        src_i3 = src_i1 + self.dropout2(src_i2)
        src_i3 = self.normi2(src_i3)

        return src_i3, src_l3

    def forward_pre(self, 
                    src_i, src_l, 
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        q0 = src_l
        k0 = src_i
        v0 = src_i
        v0 = self.norml1(v0)
        src_i0 = self.self_attn_i(q0, k0, value=v0, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_l1 = src_l + self.dropout1(src_i0)
        src_l1 = self.norml2(src_l1)
        src_l2 = self.linearl2(self.dropout(self.activation(self.linearl1(src_l1))))
        src_l3 = src_l1 + self.dropout2(src_l2)

        q1 = src_i
        k1 = src_l
        v1 = src_l
        v1 = self.normi1(v1)
        src_l0 = self.self_attn_l(q1, k1, value=v1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src_i1 = src_i + self.dropout1(src_l0)
        src_i1 = self.normi2(src_i1)
        src_i2 = self.lineari2(self.dropout(self.activation(self.lineari1(src_i1))))
        src_i3 = src_i1 + self.dropout2(src_i2)

        return src_i3, src_l3

    def forward(self, src_i, src_l):
        if self.normalize_before:
            return self.forward_pre(src_i, src_l)
        return self.forward_post(src_i, src_l)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return CWCTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
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