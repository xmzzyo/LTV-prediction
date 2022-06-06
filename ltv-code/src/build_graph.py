import os
import pickle
from multiprocessing import Pool, cpu_count

import faiss
from dtw import accelerated_dtw
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from tqdm import tqdm

from data_prepare import get_dataset
from utils import data_folder


def dtw_process(sid, s, ts_x):
    print("start process...")
    dtw_m = np.zeros((len(ts_x), len(ts_x)))
    for i in s:
        for j in range(sid + 1, len(ts_x)):
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(i, ts_x[j], dist="euclidean")
            dtw_m[sid, j] = d
            dtw_m[j, sid] = d
    return dtw_m


def pair_dtw(ts_x):
    batch_size = int(len(ts_x) / cpu_count()) + 1
    pool = Pool(processes=cpu_count())
    res = []
    dtw_m = np.zeros((len(ts_x), len(ts_x)))
    for i in range(0, len(ts_x), batch_size):
        res.append(
            pool.apply_async(dtw_process, args=(i * batch_size, ts_x[i * batch_size:(i + 1) * batch_size], ts_x,)))
    for r in res:
        dtw_m += r.get()
    pickle.dump(dtw_m, open("../data/dtw_manhattan.pickle", "wb"))
    return dtw_m


def sub_process(x, y, beta=0.5, topk=10):
    print("start precess...")
    inds = []
    if len(x) == 1:
        x = x.reshape(1, -1)
    print(x.shape, y.shape)
    dist = -beta * pair(x, y) ** 2
    for i in tqdm(range(x.shape[0]), total=x.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    return inds


def construct_graph(features, graph_path, method='heat', topk=10):
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open(graph_path, 'w')
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()

    # batch_size = 1000
    # idx = 0
    #
    # if method == 'heat':
    #     beta = 0.5
    #
    #     for i in range(int(len(features) / batch_size) + 1):
    #         print("build sub-graph", i)
    #         dist = pair(features[i * batch_size:min(len(features) - 1, (i + 1) * batch_size)], features, n_jobs=-1)
    #         dist = -beta * dist ** 2
    #         dist = np.exp(dist)
    #         inds = []
    #         for j in range(dist.shape[0]):
    #             ind = np.argpartition(dist[j, :], -(topk + 1))[-(topk + 1):]
    #             inds.append(ind)
    #         with open(fname, "a") as f:
    #             for k, v in enumerate(inds):
    #                 for vv in v:
    #                     if vv == k:
    #                         pass
    #                     else:
    #                         f.write('{} {}\n'.format(k + idx, vv))
    #         idx += batch_size
    #     print("done...")
    #
    # elif method == 'cos':
    #     features[features > 0] = 1
    #     dist = np.dot(features, features.T)
    # elif method == 'ncos':
    #     features[features > 0] = 1
    #     features = normalize(features, axis=1, norm='l1')
    #     print("get normalized")
    #     dist = np.dot(features, features.T)
    #     print("get dist")

    # inds = []
    # for i in range(dist.shape[0]):
    #     ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
    #     inds.append(ind)

    # f = open(fname, 'w')
    # A = np.zeros_like(dist)
    # for i, v in enumerate(inds):
    #     mutual_knn = False
    #     for vv in v:
    #         if vv == i:
    #             pass
    #         else:
    #             f.write('{} {}\n'.format(i, vv))
    # f.close()
    print("built graph done...")
    # edge_weight(ts_x)


def edge_weight(ts_x):
    weights = []
    with open("../data/graph/graph.txt", "r") as f:
        lines = f.readlines()
    ts_x = normalize(ts_x, axis=1, norm='l1')
    for line in tqdm(lines, total=len(lines)):
        edge = list(map(int, line.split(" ")))
        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(ts_x[edge[0]], ts_x[edge[1]], dist="euclidean")
        weights.append(d)
    with open("../data/graph_weights.txt", "w") as f:
        f.write("\n".join(list(map(str, weights))))
    print("built graph weights done...")


def build_graph(features, graph_path, k=11):
    features = features.numpy()
    d = features.shape[-1]
    nb = nq = features.shape[0]
    index = faiss.IndexFlatL2(d)
    print(index.is_trained)
    index.add(features)
    print(index.ntotal)
    D, I = index.search(features, k)  # actual search
    print(D.shape, I.shape)
    # graph_path = os.path.join(data_folder, "graphs", graph_path)
    with open(graph_path, 'w') as f:
        for i, d in enumerate(I):
            for j in d:
                if i == j:
                    continue
                f.write(f"{i} {j}\n")
                # pass


if __name__ == "__main__":
    build_graph("0")
