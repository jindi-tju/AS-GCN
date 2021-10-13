"""
Copied from https://github.com/tkipf/gcn/blob/master/gcn/utils.py

"""

import json
import pickle as pkl
import sys
from itertools import combinations
from pathlib import Path
import gensim

from gensim.scripts.glove2word2vec import glove2word2vec
import networkx as nx
import numpy as np
import scipy.sparse as sp
from gensim.models import KeyedVectors
from scipy.sparse.linalg.eigen.arpack import eigsh
import torch


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_gcn_data(dataset_str, root_dir="."):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :param root_dir: Dataset dir
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dataset_p = Path(root_dir) / "gcn_data"
    for i in range(len(names)):
        with open(dataset_p / f"ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataset_p / f"ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def load_new_data(dataset_str, root_dir="."):
    dataset_path = Path(root_dir) / "new_data" / f"{dataset_str}"
    g = nx.DiGraph()
    node2feature = {}
    node2label = {}
    with open(dataset_path / "out1_graph_edges.txt") as f:
        for line in f:
            if "node_id" in line:
                continue
            else:
                s, t = line.strip().split("\t")
                s, t = int(s), int(t)
                g.add_edge(s, t)
    with open(dataset_path / "out1_node_feature_label.txt") as f:
        for line in f:
            if "node_id" in line:
                continue
            else:
                node_id, feature, label = line.strip().split("\t")
                node_id, feature, label = int(node_id), np.array(feature.split(','), dtype=np.uint8), int(label)
                if dataset_str == "film":
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[feature] = 1
                    feature = feature_blank
                node2feature[node_id] = feature
                node2label[node_id] = label
                g.add_node(node_id, feature=feature, label=label)
    features = np.array([node2feature[i] for i in sorted(g.nodes())])
    labels = np.array([node2label[i] for i in sorted(g.nodes())])
    return g, node2feature, node2label, features, labels

def load_word_data(dataset_str, voca_token2id, root_dir=".", verbose=False):
    dataset_path = Path(root_dir) / "word_data" / f"{dataset_str}"
    g = nx.Graph()
    nodes = []
    if verbose:
        print("loading nodes.txt")
    with open(dataset_path / "nodes.txt") as f:
        for line in f:
            nodes.append(line.strip())
    node2id = {n: i for i, n in enumerate(nodes)}
    print(len(node2id))
    if verbose:
        print("loading edges.txt")
    count_true_edge = 0
    with open(dataset_path / "edges.txt") as f:
        for line in f:
            s, t = line.strip().split("\t")
            if s in node2id and t in node2id:
                g.add_edge(node2id[s], node2id[t])
                g.add_edge(node2id[t], node2id[s])
                count_true_edge = count_true_edge + 1
    print("count_true_edge:" + "\t" + str(count_true_edge))

    if verbose:
        print("loading features.txt")
    with open(dataset_path / "features.txt") as f:
        feature_lines = f.readlines()

    if verbose:
        print("loading TF-features.txt")
    with open(dataset_path / "TF-features.txt") as f:
        TF_feature_lines = f.readlines()

    if verbose:
        print("loading TF-IDF-features.txt")
    with open(dataset_path / "TF-IDF-features.txt") as f:
        TF_IDF_feature_lines = f.readlines()

    if verbose:
        print("loading labels.txt")
    with open(dataset_path / "labels.txt") as f:
        label_lines = f.readlines()
        label_lines = [label.strip() for label in label_lines]
    digit_label = label_lines[0].isdigit()
    if not digit_label:
        label_set = list(set(label_lines))
        id2label = {id: label for id, label in enumerate(label_set)}
        label2id = {label: id for id, label in enumerate(label_set)}
        with open(dataset_path / "label_id2name.json", "w") as f:
            tmp = {"id2name": id2label, "name2id": label2id}
            json.dump(tmp, f, indent=2)
    if verbose:
        print("generating node graph.txt")

    for line, label in zip(TF_feature_lines, label_lines):
        # for line, label in zip(feature_lines, label_lines):
        node, feature = line.strip().split("\t")
        if digit_label:
            label = int(label)
        else:
            label = label2id[label]
        node_id, feature = node2id[node], np.array(feature.split(' '), dtype=np.float32)
        g.add_node(node_id, feature=feature, label=label)
        # g.add_node(node_id, feature=node_representation_new[node_id].astype('float'), label=label)

    if verbose:
        print("generating word graph.txt")

    wg = nx.Graph()
    # glove embedding
    glove_input_file = dataset_path / 'glove_embedding.txt'
    word2vec_output_file = dataset_path / 'glove_model.txt'
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
    # print(count, '\n', dimensions)
    print("glove embedding success")
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # with open(dataset_path / "word_nodes.txt") as f:
    #     words = f.readlines()
    #     words = [w.strip() for w in words]
    #     word2id = {w: i for i, w in enumerate(words)}

    zero_emb = np.zeros(embeddings.vectors.shape[1], dtype=np.float32)

    if verbose:
        print("generating word graph.txt")

    for w, w_id in voca_token2id.items():
        if w not in embeddings:
            wg.add_node(w_id, feature=embeddings["<unk>"])
            continue
        else:
            wg.add_node(w_id, feature=embeddings[w])
    print("word_nodes" + "\t" + str(len(wg.nodes)))

    edge_tol = 0
    with open(dataset_path / "word_links.txt") as f:
        for line in f:
            s, t = line.strip().split("\t")
            s_id, t_id = voca_token2id[s], voca_token2id[t]
            wg.add_edge(s_id, t_id)
            wg.add_edge(t_id, s_id)
            edge_tol = edge_tol + 1
    print("edge_total" + "\t" + str(edge_tol))
    return g, wg


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


if __name__ == '__main__':
    pass
