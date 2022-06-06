import torch
from torch import nn


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlinearity=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = getattr(nn, nonlinearity)() if nonlinearity else None
        self.reset_parameters()

    def forward(self, x):
        x = self.dropout(self.fc(x))
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dims, out_dim=1, h_active="LeakyReLU", o_active="ReLU"):
        super(MLP, self).__init__()
        layers = [Dense(inp_dim, hidden_dims[0], nonlinearity=h_active)]
        for i in range(len(hidden_dims) - 1):
            layers.append(Dense(hidden_dims[i], hidden_dims[i + 1], nonlinearity=h_active))
        layers.append(Dense(hidden_dims[-1], out_dim, nonlinearity=o_active))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)
