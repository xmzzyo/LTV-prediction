import torch
from sklearn.cluster import KMeans
from torch import nn
import torch.nn.functional as F
from attention import Attention
from modules import Dense, MLP
from utils import get_device


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def assign_prob(sim, v):
    q = 1.0 / (1.0 + torch.pow(sim, 2) / v)
    q = q.pow((v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    p = target_distribution(q)
    return p


def js_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


class RegLoss(nn.Module):
    def __init__(self, te_dim, se_dim, k, k_dim):
        super(RegLoss, self).__init__()
        self.centroids = nn.Parameter(torch.randn((k, k_dim)))
        nn.init.orthogonal_(self.centroids.data)
        self.v = 1
        self.t_bilinear = Dense(te_dim, k_dim, nonlinearity="Tanh")
        self.s_bilinear = Dense(se_dim, k_dim, nonlinearity="Tanh")

    def forward(self, te, se):
        te = torch.mean(te, dim=0)
        t_prob = assign_prob(torch.tanh(self.t_bilinear(te).mm(self.centroids.transpose(1, 0))), self.v)
        s_prob = assign_prob(torch.tanh(self.s_bilinear(se).mm(self.centroids.transpose(1, 0))), self.v)
        return js_div(t_prob, s_prob)


if __name__ == "__main__":
    uid = torch.randint(0, 10000, (16,))
    wave_coeffs = torch.rand(2, 16, 100)
    s_l = torch.randn(10000, 100)
    xnext = torch.randn(16, 100)
    anext = torch.randn(100, 100)
    net = RegLoss(100, k=30)
    loss = net(wave_coeffs, xnext)
    print(loss)
