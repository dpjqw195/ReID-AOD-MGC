import torch.nn as nn
import torch
from common import MultiHeadAttention,PositionwiseFeedForward,PositionalEncoding
from torch.nn import functional as F
from torch.nn import init
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_input1 = F.max_pool2d(enc_input,[2,1])
        enc_input2 = F.max_pool2d(enc_input,[4,1])
        enc_output1, _ = self.slf_attn1(enc_input2,enc_input1,enc_input1)
        enc_output2, enc_slf_attn = self.slf_attn2(enc_input,enc_output1,enc_output1)
        return enc_output2, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner,  dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, x, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        if self.scale_emb:
            x *= self.d_model ** 0.5
        # x = self.dropout(self.position_enc(x))
        # x = self.layer_norm(x)

        for enc_layer in self.layer_stack:
            x, enc_slf_attn = enc_layer(x, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return x, enc_slf_attn_list
        return x

