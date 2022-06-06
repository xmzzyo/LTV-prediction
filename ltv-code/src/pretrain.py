import os
import time

import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dgl import graph as dgl_graph

from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import generate_mask_tensor
from dgl.nn import GATConv
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BatchNorm1d, ReLU
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gcn import get_gnnp_data
from metrics import NMSE, NMAE, MAE, MSE
from utils import data_folder, ReLU1000, get_device


class GraphDataset(DGLDataset):
    def __init__(self,
                 data_name,
                 limit_num,
                 future_length=90,
                 train_ratio=1.0,
                 k=31,
                 save_dir=None,
                 force_reload=False,
                 verbose=True):
        self.data_name = data_name
        self.limit_num = limit_num
        self.future_length = future_length
        self.train_ratio = train_ratio
        self.k = k
        super().__init__(name='dgl_data',
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def process(self):
        self._g = self._load_graph()
        if self.verbose:
            print('Finished data loading and preprocessing.')
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumTrainingSamples: {}'.format(
                dgl.backend.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                dgl.backend.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                dgl.backend.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def _load_graph(self):
        feats, edge_index, labels = get_gnnp_data(self.data_name, self.k, self.limit_num, self.future_length,
                                                  self.train_ratio)
        edge_index = torch.tensor(edge_index).T
        edge_index = edge_index[0], edge_index[1]
        g = dgl.graph(edge_index, idtype=torch.int32)
        g = dgl.to_bidirected(g)
        g.ndata['feat'] = feats
        g.ndata['label'] = labels

        idx = np.arange(labels.shape[0])
        idx_train, idx_test = train_test_split(idx, test_size=0.4, random_state=2021)
        idx_test, idx_val = train_test_split(idx_test, test_size=0.5, random_state=2021)

        train_mask = self._sample_mask(idx_train, labels.shape[0])
        val_mask = self._sample_mask(idx_val, labels.shape[0])
        test_mask = self._sample_mask(idx_test, labels.shape[0])
        g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)
        return g

    def _sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return mask

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  f'dgl_graph_{self.data_name}_{self.future_length}_{self.train_ratio}.bin')
        save_graphs(str(graph_path), self._g)

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  f'dgl_graph_{self.data_name}_{self.future_length}_{self.train_ratio}.bin')
        return os.path.exists(graph_path)

    def load(self):
        graph = load_graphs(os.path.join(self.save_path, f'dgl_graph_{self.future_length}_{self.train_ratio}.bin'))
        self._g = graph[0]
        if self.verbose:
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumTrainingSamples: {}'.format(
                dgl.backend.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                dgl.backend.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                dgl.backend.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], out_dim, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        self.bn = BatchNorm1d(in_dim)

    def forward(self, inputs):
        inputs = inputs.to(get_device())
        h = self.bn(inputs)
        for l in range(self.num_layers):
            # print(f"h shape: {h.shape}")
            h = self.gat_layers[l](self.g, h).flatten(1)
        # print(f"h shape: {h.shape}")
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        # print(f"logits: {logits.shape}")
        return logits

