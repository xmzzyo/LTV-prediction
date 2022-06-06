import random

import torch
from torch import nn
from torch.nn import BatchNorm1d

from attention import Attention
from modules import Dense, MLP
from utils import init_rnn


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, te_dim, se_dim, out_dim=100):
        super(MLPDecoder, self).__init__()
        att_head_num = 4
        self.attn = Attention(att_head_num, te_dim, hidden_dim, hidden_dim, hidden_dim, dropout=0.1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feat_fusion = GateFusion(te_dim, se_dim)
        self.hidden_fc = Dense(te_dim, out_dim, nonlinearity="Tanh")
        self.gnn_fc = Dense(se_dim, out_dim, nonlinearity="Tanh")
        # self.te_bn = BatchNorm1d(out_dim)
        self.se_bn = BatchNorm1d(se_dim)
        self.residual = True
        if self.residual:
            concat_size = out_dim + 1
        # self.is_active = MLP(concat_size, [300, 100, 50], h_active="LeakyReLU", o_active="Sigmoid")
        self.ltv = MLP(concat_size, [300, 100, 50], h_active="LeakyReLU", o_active="ReLU")

    def forward(self, prev_y, hidden, gnn_embed):
        if hidden is not None:
            # [num_layers * num_directions, batch, hidden_size] * (level+1)
            hidden = torch.cat(hidden, dim=0).permute(1, 0, 2)
            attn_output, p_attn = self.attn(hidden, hidden, hidden)
            attn_output = self.avg_pool(attn_output.transpose(1, 2)).squeeze(-1)
            # attn_output = self.te_bn(attn_output)
            if gnn_embed is not None:
                gnn_embed = self.se_bn(gnn_embed)
                # gnn_embed = self.gnn_fc(gnn_embed)
                hidden = self.feat_fusion(attn_output, gnn_embed)
            else:
                hidden = attn_output
            hidden = self.hidden_fc(hidden)
        elif gnn_embed is not None and hidden is None:
            hidden = self.gnn_fc(gnn_embed)

        if self.residual:
            hidden = torch.cat([hidden, prev_y.unsqueeze(1)], dim=-1)
        ltv = self.ltv(hidden)
        # active = self.is_active(hidden)
        return ltv.squeeze()


class GateFusion(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(GateFusion, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, y):
        r_f = torch.cat([x, y], dim=-1)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * x
        return o


# Deprecated
class RNNDecoder(nn.Module):
    def __init__(self, inp_levels, inp_dim, hidden_dim, kv_sizes=[15, 7, 3, 3], nonlinearity='SELU', residual=True,
                 attn_heads=None, attn_size=None, dropout=0.1, rnn_type='GRU', n_layers=1):
        super(RNNDecoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(inp_dim, hidden_dim, num_layers=n_layers,
                                         dropout=dropout, batch_first=True, bidirectional=True)
        self.residual = residual
        concat_size = 2 * hidden_dim
        if attn_heads is not None:
            self.attn = HierarchicalAttn(inp_levels, 4, attn_size, 2 * hidden_dim, kv_sizes, dropout=dropout)
            concat_size += attn_size
        if residual:
            concat_size += inp_dim
        self.y_output = MLP(concat_size, [200, 100, 50], h_active="PReLU", o_active="ReLU")
        self.is_terminate = MLP(concat_size, [200, 100, 50], h_active="PReLU", o_active="Sigmoid")
        init_rnn(self.rnn, 'xavier')

    def forward(self, prev_y, enc_compress, hidden):
        # print(prev_y.shape)
        hidden = hidden.contiguous()
        output, hidden = self.rnn(prev_y, hidden)
        if hasattr(self, 'attn'):
            attn_output, p_attn = self.attn(output, enc_compress)
        else:
            attn_output, p_attn = None, None
        # print(output.shape, attn_output.shape, prev_y.shape)
        concat = torch.cat([output, attn_output, prev_y if self.residual else None], dim=-1)
        next_y = self.y_output(concat)
        next_state = self.is_terminate(concat)
        return next_y, next_state, output, hidden, p_attn


class CLDRNNDecoder(nn.Module):
    def __init__(self, enc_levels, time_slice, hidden_dim=100, gnn_dim=100):
        super(CLDRNNDecoder, self).__init__()
        max_lens = {"T": 1, "M": 3, "W": 13, "D": 90}
        self.max_len = max_lens[time_slice]
        # aggregate multi-level rnn's hidden
        self.hidden_mlp = MLP(2 * enc_levels, [30, 50, 30], out_dim=2, o_active="SELU")
        self.hidden_fusion = GateFusion(hidden_dim, gnn_dim)
        self.decoder = RNNDecoder(enc_levels, 1, hidden_dim, attn_heads=4, attn_size=2 * hidden_dim)
        self.teacher_forcing_ratio = 0.2

    def forward(self, prev_y, enc_compress, hiddens, gnn_embed, y):
        hiddens = torch.cat(hiddens, dim=0).permute(1, 2, 0)
        hidden = self.hidden_mlp(hiddens)
        if gnn_embed is not None:
            gnn_embed = gnn_embed.unsqueeze(0).expand(2, -1, -1)
            hidden = self.hidden_fusion(hidden.permute(2, 0, 1), gnn_embed)
        else:
            hidden = hidden.permute(2, 0, 1)

        teacher_y = y

        ys = None
        mask_pred = None
        for i in range(self.max_len):
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            if use_teacher_forcing and teacher_y is not None:
                prev_y = teacher_y[:, i].view(teacher_y.size(0), 1, 1)
            prev_y, next_state, output, hidden, p_attn = self.decoder(prev_y, enc_compress, hidden)
            if ys is None:
                ys = prev_y.squeeze(1)
            else:
                ys = torch.cat([ys, prev_y.squeeze(1)], dim=1)
            if mask_pred is None:
                mask_pred = next_state.squeeze(1)
            else:
                mask_pred = torch.cat([mask_pred, next_state.squeeze(1)], dim=1)
        return ys, mask_pred


class HierarchicalAttn(nn.Module):
    def __init__(self, inp_levels, attn_heads, attn_size, query_size, kv_sizes=[15, 7, 3, 3], dropout=0.1):
        super(HierarchicalAttn, self).__init__()
        # self.mlp = MLP(inp_levels, hidden_dims=[30, 20, 10])
        # self.channels_query = [nn.Parameter(torch.randn((1, query_size))) for _ in range(inp_levels)]
        self.attns = nn.ModuleList(
            [Attention(attn_heads, attn_size, query_size, query_size, query_size, dropout=dropout) for kv_size in
             kv_sizes])
        self.level_attn = Attention(attn_heads, attn_size, query_size, query_size, query_size, dropout=dropout)

    def forward(self, dec_out, enc_compresses):
        encoder_attented = []
        for enc_compress, attn in zip(enc_compresses, self.attns):
            # print(dec_out.shape, enc_compress.shape)
            values, weights = attn(dec_out, enc_compress, enc_compress)
            # print(values.shape, weights.shape)
            encoder_attented.append(values)
        encoder_attented = torch.cat(encoder_attented, dim=1)
        # print(dec_out.shape, encoder_attented.shape)
        out, weights = self.level_attn(dec_out, encoder_attented, encoder_attented)
        # print(out.shape, weights.shape) torch.Size([8, 1, 100]) torch.Size([8, 2, 1, 100])
        return out, weights
