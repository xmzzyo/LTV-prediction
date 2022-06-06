import os
import pickle

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import nn
from torch.optim import Adam

from build_graph import construct_graph, build_graph
from data_prepare import get_dataset
from dual_loss import target_distribution
from kmeans import kmeans
from mincut_pool import dense_mincut_pool
from modules import MLP, Dense
from utils import get_device


class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        output = F.relu(output)
        output = F.dropout(output, training=self.training)
        return output


class GNNP(nn.Module):
    def __init__(self, in_dim, out_dim=100, hidden_dims=[200, 300, 150]):
        super(GNNP, self).__init__()
        self.gnn_1 = GNNLayer(in_dim, hidden_dims[0])
        self.gnn_2 = GNNLayer(hidden_dims[0], hidden_dims[1])
        self.gnn_3 = GNNLayer(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], out_dim)

    def forward(self, feat_x, adj):
        h = self.gnn_1(feat_x, adj)
        h = self.gnn_2(h, adj)
        h = self.gnn_3(h, adj)
        return F.relu(self.out(h))


def get_adj_m(nums, edge_index):
    adj_m = torch.zeros((nums, nums))
    for edge in edge_index:
        adj_m[edge[0], edge[1]] = 1
    return adj_m.to(get_device())


def get_gnnp_data(data_name, k=11, limit_num=-1, future_len=90, train_ratio=1.0):
    ts_x, feat_x, y = get_dataset(data_name, limit_num=limit_num, future_len=future_len)
    if limit_num > 0:
        graph_path = f'../data/graphs/ol_graph_{data_name}_{limit_num}_{train_ratio}.txt'
    else:
        graph_path = f'../data/graphs/ol_graph_{data_name}_{ts_x.shape[0]}_{train_ratio}.txt'
    features = torch.cat([ts_x, feat_x], dim=-1)
    if not os.path.exists(graph_path):
        print(f"building knn graph for {data_name}...")
        # construct_graph(feat_x.numpy(), graph_path, method='heat', topk=10)
        build_graph(features, graph_path, k)
    with open(graph_path, "r") as f:
        lines = f.readlines()
    edge_index = []
    for line in lines:
        edge = list(map(int, line.split(" ")))
        edge_index.append(edge)
    return features, edge_index, y


class PTGNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PTGNN, self).__init__()
        self.embed = GNNP(in_dim, out_dim=out_dim, hidden_dims=[200, 300, 200])
        self.out = Dense(out_dim, 1, nonlinearity="ReLU")

    def forward(self, data, adj_m):
        z_l = self.embed(data, adj_m)
        return self.out(z_l).squeeze(-1)


class GCNClusterP(nn.Module):
    def __init__(self, in_dim, out_dim, cluster_k):
        super(GCNClusterP, self).__init__()
        self.cluster_k = cluster_k
        self.embed = GNNP(in_dim, out_dim=out_dim, hidden_dims=[200, 300, 200])
        self.bilinears = Dense(out_dim, out_dim, nonlinearity="ReLU")
        self.v = 1

    def forward(self, data, adj_m, cluster=False):
        z_l = self.embed(data, adj_m)
        clu_loss = None
        if cluster:
            clustering = MiniBatchKMeans(n_clusters=self.cluster_k, max_iter=1000)
            clustering.fit(z_l.cpu().detach().numpy())
            cluster_centers = torch.tensor(clustering.cluster_centers_).to(get_device())
            print("clustering done...")
            #
            # print(z_l.shape, cluster_centers.shape, (z_l.unsqueeze(1) - cluster_centers).shape)
            #
            # q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - cluster_centers, 2), 2) / self.v)
            q = 1.0 / (1.0 + torch.pow(torch.tanh(self.bilinears(z_l).mm(cluster_centers.transpose(1, 0))), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            p = target_distribution(q)
            #
            # s_l = q
            # s_l = self.assign_mat(z_l)
            # xnext, out_adj, mincut_loss, ortho_loss = dense_mincut_pool(z_l, adj_m, s_l)

            # s_l = F.softmax(torch.matmul(z_l, cluster_centers.permute(1, 0)), dim=-1)
            # s_l = F.softmax(self.assign_mat(data, adj_m), dim=-1)
            # s_l = F.softmax(self.assign_mat(z_l), dim=-1)
            xnext = torch.matmul(q.transpose(-1, -2), z_l)
            # anext = (s_l.transpose(-1, -2)).matmul(adj_m).matmul(s_l)

            clu_loss = F.kl_div(q.log(), p, reduction='batchmean')
            # print("clu loss: ", clu_loss.item())
            return z_l, p, xnext.squeeze(0), clu_loss
        else:
            return z_l, None, None, clu_loss


def pretrain_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, adj, y = get_gnnp_data(data_name="0", limit_num=30000)
    features = features.to(device)
    adj = adj.to(device)
    y = y.to(device)
    gnn = PTGNN(213, 50).to(device)
    optimizer = Adam(gnn.parameters(), lr=1e-4)
    # train
    gnn.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = gnn(features, adj)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, loss: {loss}")


if __name__ == "__main__":
    pretrain_gnn()
    exit()
    data, adj_m = get_gnnp_data()
    print(data.shape, adj_m.shape)
    # clustering = KMeans(n_clusters=100, n_init=20).fit(data.numpy())
    # cluster_ids_x, cluster_centers = kmeans(X=data, num_clusters=100, distance='cosine', device=get_device())
    # print(cluster_centers.shape)
    # exit()
    gnn = GCNClusterP(213, 50, cluster_k=100)
    out = gnn(data, adj_m, True)
    for o in out:
        print(o.shape)
