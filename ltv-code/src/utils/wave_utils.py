# -*- coding: utf-8 -*-

"""
Create distribution tensors for wavelet filter operations and -regularisation.
"""

# Basic import(s)
import math

import numpy as np
from scipy.ndimage.interpolation import shift

import torch

from utils import get_device


def dist_low_(N, m):
    """
    Create distribution tensor `D`, such that matrix product of filter
    coefficients `a` and `D` yield the low-pass filter matrix, `a * D = L`.
    2^m -> 2^(m-1)

    Arguments:
        N: Number of filter coefficients.
        m: Dimension (radix-2) matrix, column-wise.

    Returns:
        Low-pass filter distribution tensor, as np.array.
    """
    L = np.zeros((N, 2 ** (m - 1), 2 ** m))
    for i in range(N):
        for j in range(L.shape[1]):
            L[i, j, int(2 * j - i + N / 2) % L.shape[2]] = 1
    return np.transpose(L, [1, 0, 2])


def dist_high_(N, m):
    """
    Create distribution tensor `D`, such that matrix product of filter
    coefficients `a` and `D` yield the high-pass filter matrix, `a * D = H`.
    2^m -> 2^(m-1)

    Arguments:
        N: Number of filter coefficients.
        m: Dimension (radix-2) matrix, column-wise.

    Returns:
        High-pass filter distribution tensor, as np.array.
    """
    H = np.zeros((N, 2 ** (m - 1), 2 ** m))
    for i in range(N):
        for j in range(H.shape[1]):
            H[i, j, int(2 * j + i - N / 2 + 1) % H.shape[2]] = (-1) ** i
    return np.transpose(H, [1, 0, 2])


def dist_reg_(N):
    """
    Create distribution tensor `D` for regularisation terms (R3-5), such that
    each term is a row in the resulting matrix `a * D * a`, e.g.

    Arguments:
        N: Number of filter coefficients.

    Returns:
        Regularisation distribution tensor, as torch.autograd.Variable.
    """

    D = np.zeros((N, N - 1, N))
    for idx in range(N):
        D[idx, :, idx] = 1
        for row, m in enumerate(range(-N // 2 + 1, N // 2)):
            D[idx, row, :] = shift(D[idx, row, :], 2 * m)
    D = np.transpose(np.round(D).astype(np.int), [1, 0, 2])
    return torch.FloatTensor(D).to(get_device())


def delta_(N):
    """
    Vector Kronecker-delta, with `1` on central entry.

    Arguments:
        N: Number of vector entries, required to be odd.

    Return:
        Vector Kronecker-delta, as torch.autograd.Variable.
    """
    # Check(s)
    assert N % 2 == 1, "delta_: Number of entries ({}) has to be odd.".format(N)

    # Kronecker delta
    delta = torch.zeros(N).to(get_device())
    delta[N // 2] = 1
    return delta


def init_wave_filter(params):
    """
    Transform initial parameter configuration to conform with as many
    constraints as possible, so as to avoid exploding gradients.

    Arguments:
        params: Initial, random parameter configuration, as array of size (N,).

    Returns
        Transformed `params`.
    """

    # Definitions
    N = len(params)

    # Utility functions
    # -----------------
    # 2-norm of vector `x`.
    norm_ = lambda x: np.sqrt(np.sum(np.square(x)))

    # Get high-pass filter coefficients `b` from low-pass filter coefficients `a`.
    b_ = lambda a: np.flip(a, 0) * np.power(-1., np.arange(N))

    # Constraint functions
    # --------------------
    # Add constant to low-pass filter coefficients `a` so as to satisfy (C1).
    c1_ = lambda a: a + (1. - np.sum(a)) / float(len(a))

    # Divide low-pass filter coefficients `a` by constant so as to satisfy (C2, m=0).
    c2_ = lambda a: a / norm_(a)

    # Add constant to high-pass filter coefficients `b` so as to satisfy (C4).
    c4_ = lambda a: a - b_(np.repeat(- np.sum(b_(a)) / float(N), N))

    # Iteratively enforce each of the above conditions
    for _ in range(10):
        params = c1_(params)
        params = c2_(params)
        params = c4_(params)

    return params


class Regularisation(torch.nn.Module):
    def __init__(self, coeff_len):
        """Wavelet-regularisation for filter coefficient `params`."""
        super(Regularisation, self).__init__()
        self.D = dist_reg_(coeff_len)  # Distribution tensor
        self.delta = delta_(coeff_len - 1)  # Kronecker-delta

    def forward(self, lo, hi):
        # Compute high-pass filter coefficients
        # hi = coeff[::-1]  # Reverse `a`
        # sign = torch.FloatTensor(np.power(-1, np.arange(self.N))).to(device)
        # hi = hi * sign  # Flip sign of every other element in `b`
        lo.squeeze_()
        hi.squeeze_()
        # (R1)
        R1 = torch.pow(lo.sum() - np.sqrt(2.), 2.)

        # (R2)
        R2 = torch.pow(lo.matmul(self.D).matmul(lo) - self.delta, 2.).sum()

        # (R3)
        R3 = torch.pow(hi.matmul(self.D).matmul(hi) - self.delta, 2.).sum()

        # (R4)
        R4 = torch.pow(hi.sum(), 2.)

        # (R5)
        R5 = torch.pow(lo.matmul(self.D).matmul(hi), 2.).sum()

        return R1 + R2 + R3 + R4 + R5


def recon_coeff(yl, yh):
    ll = yl.squeeze()
    # Do a multilevel inverse transform
    for h in yh[::-1]:
        if h is None:
            h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                            ll.shape[-1], device=ll.device)
        h = h.squeeze()
        # 'Unpad' added dimensions
        if ll.shape[-2] > h.shape[-2]:
            ll = ll[..., :-1, :]
        if ll.shape[-1] > h.shape[-1]:
            ll = ll[..., :-1]
        lh, hl, hh = torch.unbind(h, dim=1)
        ll_col1 = torch.cat([ll, lh], dim=1)
        ll_col2 = torch.cat([hl, hh], dim=1)
        ll = torch.cat([ll_col1, ll_col2], dim=2)
    return ll


def losses(self, ll, yh):
    c = recon_coeff(ll, yh)
    lambda_reg = 1.0E+02
    # Sparsity loss
    sparsity = 1. - self.gini(c)

    # Regularisation loss
    regularisation = self.col_reg.forward(self.lo_col.view(1, -1), self.hi_col.view(1, -1)) + \
                     self.row_reg.forward(self.lo_row.view(1, -1), self.hi_row.view(1, -1))

    # Compactness  loss
    compactness_col = torch.sum(
        torch.dot(self.indices - 0.5, self.lo_col.view(self.lo_col.numel()).abs() / self.lo_col.abs().sum()))
    compactness_row = torch.sum(
        torch.dot(self.indices - 0.5, self.lo_row.view(self.lo_row.numel()).abs() / self.lo_row.abs().sum()))
    # Combined loss
    combined = sparsity + lambda_reg * regularisation + compactness_col + compactness_row
    return combined
