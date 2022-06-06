import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from decoder import GateFusion
from utils import get_device, init_rnn
from utils.wave_utils import init_wave_filter


class MultiChannelWave(nn.Module):
    def __init__(self, channels, filter_len, levels, seq_len, rnn_type, hid_dim=50, out_dim=50, drop_out=0.2):
        super(MultiChannelWave, self).__init__()
        self.channels = channels
        self.filter_len = filter_len
        self.levels = levels
        # TODO init with Morlet bases
        self.filters = self._init_wave_filters()
        self.wave_decs = nn.ModuleList([WaveDec(seq_len, l, h) for l, h in self.filters])
        self.te_encoders = nn.ModuleList(
            [RNNEncoder(inp_dim=self.channels, hidden_dim=hid_dim, out_dim=out_dim, dropout=drop_out,
                        rnn_type=rnn_type) for _ in range(levels + 1)])
        # self.gate_fusion = nn.ModuleList(
        #     [GateFusion(input_size=channels, fusion_size=50) for _ in range(levels + 1)])

    def forward(self, inputs):
        # each layer: [batch_size * down_sample_size * channels]
        level_coeffs = [None for _ in range(2 * self.levels)]
        # iter channels
        for wave_dec in self.wave_decs:
            # xh_1, xl_1, xh_2, xl_2, xh_3, xl_3
            dec_coeffs = wave_dec(inputs)
            for i in range(self.levels):
                if level_coeffs[2 * i] is None:
                    level_coeffs[2 * i] = dec_coeffs[2 * i]
                    level_coeffs[2 * i + 1] = dec_coeffs[2 * i + 1]
                else:
                    level_coeffs[2 * i] = torch.cat([level_coeffs[2 * i], dec_coeffs[2 * i]], dim=-1)
                    level_coeffs[2 * i + 1] = torch.cat([level_coeffs[2 * i + 1], dec_coeffs[2 * i + 1]], dim=-1)
        te_outputs = []
        te_hiddens = []
        xh, xl = None, None
        for i, te in enumerate(self.te_encoders):
            # encode first k layer's high freq
            if i < self.levels:
                xh, xl = level_coeffs[2 * i], level_coeffs[2 * i + 1]
                x = xh
            else:
                # encode last layer's low freq
                x = xl
            # if gnn_embed is not None:
            #     x = self.gate_fusion[i](x, gnn_embed.unsqueeze(1).expand(-1, x.size(1), -1))
            # print(f"{i} x shape", x.shape)
            outputs, hidden = te(x)
            # print(outputs.shape, hidden.shape)
            te_outputs.append(outputs)
            te_hiddens.append(hidden)
        return level_coeffs, te_outputs, te_hiddens

    def _init_wave_filters(self):
        filters = []
        for i in range(self.channels):
            params = np.random.randn(self.filter_len)
            params = init_wave_filter(params)
            l_filter = params[::-1]
            h_filter = params * np.power(-1, np.arange(self.filter_len))
            filters.append((l_filter, h_filter))
        return filters

    def wavelets_losses(self):
        losses = 0.0
        for wave_dec in self.wave_decs:
            losses += wave_dec.wavelets_loss()
        return losses


class WaveDec(nn.Module):
    def __init__(self, seq_len, l_filter, h_filter):
        super(WaveDec, self).__init__()
        self.seq_len = seq_len

        self.mWDN1_H = nn.Linear(seq_len, seq_len)
        self.mWDN1_L = nn.Linear(seq_len, seq_len)
        self.mWDN2_H = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.mWDN2_L = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.mWDN3_H = nn.Linear(int(seq_len / 4), int(seq_len / 4))
        self.mWDN3_L = nn.Linear(int(seq_len / 4), int(seq_len / 4))
        self.down_sample = nn.AvgPool1d(2)
        self.sigmoid = nn.Sigmoid()

        self.l_filter = l_filter
        self.h_filter = h_filter

        self.cmp_mWDN1_H = self.init_wave_matrix(seq_len, False, is_comp=True)
        self.cmp_mWDN1_L = self.init_wave_matrix(seq_len, True, is_comp=True)
        self.cmp_mWDN2_H = self.init_wave_matrix(int(seq_len / 2), False, is_comp=True)
        self.cmp_mWDN2_L = self.init_wave_matrix(int(seq_len / 2), True, is_comp=True)
        self.cmp_mWDN3_H = self.init_wave_matrix(int(seq_len / 4), False, is_comp=True)
        self.cmp_mWDN3_L = self.init_wave_matrix(int(seq_len / 4), True, is_comp=True)

        self.mWDN1_H.weight = nn.Parameter(self.init_wave_matrix(seq_len, False))
        self.mWDN1_L.weight = nn.Parameter(self.init_wave_matrix(seq_len, True))
        self.mWDN2_H.weight = nn.Parameter(self.init_wave_matrix(int(seq_len / 2), False))
        self.mWDN2_L.weight = nn.Parameter(self.init_wave_matrix(int(seq_len / 2), True))
        self.mWDN3_H.weight = nn.Parameter(self.init_wave_matrix(int(seq_len / 4), False))
        self.mWDN3_L.weight = nn.Parameter(self.init_wave_matrix(int(seq_len / 4), True))

    def forward(self, inputs):
        # input = input.view(input.shape[0], input.shape[1])
        ah_1 = self.sigmoid(self.mWDN1_H(inputs))
        al_1 = self.sigmoid(self.mWDN1_L(inputs))
        xh_1 = self.down_sample(ah_1.view(ah_1.shape[0], 1, -1))
        xl_1 = self.down_sample(al_1.view(al_1.shape[0], 1, -1))

        ah_2 = self.sigmoid(self.mWDN2_H(xl_1))
        al_2 = self.sigmoid(self.mWDN2_L(xl_1))
        xh_2 = self.down_sample(ah_2)
        xl_2 = self.down_sample(al_2)

        ah_3 = self.sigmoid(self.mWDN3_H(xl_2))
        al_3 = self.sigmoid(self.mWDN3_L(xl_2))
        xh_3 = self.down_sample(ah_3)
        xl_3 = self.down_sample(al_3)

        xh_1 = xh_1.transpose(1, 2)
        xl_1 = xl_1.transpose(1, 2)
        xh_2 = xh_2.transpose(1, 2)
        xl_2 = xl_2.transpose(1, 2)
        xh_3 = xh_3.transpose(1, 2)
        xl_3 = xl_3.transpose(1, 2)

        return xh_1, xl_1, xh_2, xl_2, xh_3, xl_3

    def inverse(self):
        pass

    def init_wave_matrix(self, seq_len, is_l, is_comp=False):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter

        max_epsilon = np.min(np.abs(filter_list))
        if is_comp:
            weight_np = np.zeros((seq_len, seq_len))
        else:
            weight_np = np.random.randn(seq_len, seq_len) * 0.1 * max_epsilon

        for i in range(0, seq_len):
            filter_index = 0
            for j in range(i, seq_len):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return torch.tensor(weight_np, dtype=torch.float).to(get_device())

    def wavelets_loss(self):
        W_mWDN1_H = self.mWDN1_H.weight
        W_mWDN1_L = self.mWDN1_L.weight
        W_mWDN2_H = self.mWDN2_H.weight
        W_mWDN2_L = self.mWDN2_L.weight
        W_mWDN3_H = self.mWDN3_H.weight
        W_mWDN3_L = self.mWDN3_L.weight
        L_loss = torch.norm((W_mWDN1_L - self.cmp_mWDN1_L), 2) + torch.norm((W_mWDN2_L - self.cmp_mWDN2_L),
                                                                            2) + torch.norm(
            (W_mWDN3_L - self.cmp_mWDN3_L), 2)
        H_loss = torch.norm((W_mWDN1_H - self.cmp_mWDN1_H), 2) + torch.norm((W_mWDN2_H - self.cmp_mWDN2_H),
                                                                            2) + torch.norm(
            (W_mWDN3_H - self.cmp_mWDN3_H), 2)
        loss = L_loss + H_loss
        return loss


class TEncoder(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=2, dropout=0.2):
        super(TEncoder, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, hidden=None):
        pass

    def get_output_dim(self):
        pass


class RNNEncoder(TEncoder):
    def __init__(self, inp_dim, hidden_dim, out_dim, nonlinearity='SELU', dropout=0.1, rnn_type='GRU', num_layers=1,
                 initializer='xavier'):
        super(RNNEncoder, self).__init__(inp_dim, out_dim)
        self.rnn = getattr(nn, rnn_type)(inp_dim, hidden_dim, num_layers=num_layers,
                                         dropout=dropout, batch_first=True, bidirectional=True)
        # self.compress = Dense(hidden_dim * 2, out_dim, dropout=dropout, nonlinearity=nonlinearity)
        init_rnn(self.rnn, initializer)

    def forward(self, x, num=None, cat=None):
        outputs, hidden = self.rnn(x)
        # compress = self.compress(outputs)
        return outputs, hidden

    def get_output_dim(self):
        return self.num_layers * 2 * self.out_dim
