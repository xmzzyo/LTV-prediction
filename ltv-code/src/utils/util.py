import json
import math
import os

import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch
from torch.nn import init, Hardtanh

data_folder = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "../.."), 'data')


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data):
    return (d.to(get_device()) for d in data)


def init_rnn(x, type='uniform'):
    for layer in x._all_weights:
        for w in layer:
            if 'weight' in w:
                if type == 'xavier':
                    init.xavier_normal_(getattr(x, w))
                elif type == 'uniform':
                    stdv = 1.0 / math.sqrt(x.hidden_size)
                    init.uniform_(getattr(x, w), -stdv, stdv)
                elif type == 'normal':
                    stdv = 1.0 / math.sqrt(x.hidden_size)
                    init.normal_(getattr(x, w), .0, stdv)
                else:
                    raise ValueError


class ReLU1000(Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU1000, self).__init__(0., 1000., inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result_avg(self):
        return dict(self._data.total)

    def result(self, weights, key):
        result = dict(self._data.average)
        mix = 0.0
        for k, v in weights.items():
            mix += v * result[k]
        result[key] = mix
        return result
