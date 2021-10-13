import pickle
import random
from pathlib import Path

import torch.nn as nn
import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
from scipy import sparse

from src.utils import load_gcn_data, load_new_data, load_word_data
from src.neural_topic_model import NTM_model

def _read_file(filepath, func, head=False):
    result = []
    with open(filepath) as f:
        if head:
            f.readline()
        for line in f:
            result.append(func(line.strip()))
    return result


def _read_node_int(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: int(x), head=head)


def _read_node_str(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: x, head=head)


def _read_edge_str(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: x.split("\t"), head=head)


def _read_edge_int(filepath, head=False):
    return _read_file(filepath=filepath, func=lambda x: [int(tmp) for tmp in x.split("\t")], head=head)


class GraphDataset(object):
    def __init__(self, args, dataset_str, bow_dictionary, existing_partition=0, raw=True, root_dir="data"):
        self.dataset_str = dataset_str
        self.existing_partition = existing_partition
        self.root_dir = Path(root_dir)
        if raw:
            self._load_dataset_raw(args=args, bow_dictionary=bow_dictionary, dataset_str=dataset_str)
        else:
            self._load_dataset_pickled(dataset_path=dataset_str)

    def _load_dataset_raw(self, args, bow_dictionary, dataset_str):
        if dataset_str in ["hep_small", "hep_large", "dblp", "cora_enrich"]:
            # Two_stage
            if args.two_stage:
                topic_represent, topic_word_adj, doc_represent, node_topic_adj, bow_dictionary = NTM_model(args)
                self.node_topic_adj = node_topic_adj
                self.topic_word_adj = topic_word_adj
                self.topic_features = topic_represent

            dg, dwg = load_word_data(dataset_str=dataset_str, voca_token2id=bow_dictionary.token2id, root_dir=self.root_dir)
            self.adj = nx.adjacency_matrix(dg, sorted(dg.nodes()))
            self.features = np.array(
                [features for _, features in sorted(dg.nodes(data='feature'), key=lambda x: x[0])])
            self.word_adj = nx.adjacency_matrix(dwg, sorted(dwg.nodes()))
            self.word_features = np.array(
                [features for _, features in sorted(dwg.nodes(data='feature'), key=lambda x: x[0])])
            self.labels = np.array([label for _, label in sorted(dg.nodes(data='label'), key=lambda x: x[0])])
            self.G = dg
            self.G_self_loop = nx.Graph(self.adj + sparse.eye(self.adj.shape[0]))
            self.node_list = sorted(self.G.nodes())
            if self.existing_partition:
                self.train_nodes = _read_node_int(self.root_dir + f"/word_data/{self.dataset_str}/nodes.train")
                self.val_nodes = _read_node_int(self.root_dir + f"/word_data/{self.dataset_str}/nodes.val")
                self.test_nodes = _read_node_int(self.root_dir + f"/word_data/{self.dataset_str}/nodes.test")
            else:
                interest_node_ids = self.node_list.copy()
                random.shuffle(interest_node_ids)
                validation_size = int(len(interest_node_ids) * 0.1)
                test_size = int(len(interest_node_ids) * 0.1)
                self.val_nodes = interest_node_ids[:validation_size]
                self.test_nodes = interest_node_ids[validation_size:(validation_size + test_size)]
                self.train_nodes = [node_id for node_id in interest_node_ids if
                                    node_id not in self.val_nodes and node_id not in self.test_nodes]

            output_pickle_file_name = self.root_dir / f"word_data/{dataset_str}/{dataset_str}.pickle.bin"
        else:
            print(f"{dataset_str} not defined")
            raise NotImplementedError
        print("start saving pickle data")

        with open(output_pickle_file_name, 'wb') as fout:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "dataset_str": self.dataset_str,
                "adj": self.adj,
                "G": self.G,
                "G_self_loop": self.G_self_loop,
                "node_list": self.node_list,
                "train_nodes": self.train_nodes,
                "val_nodes": self.val_nodes,
                "test_nodes": self.test_nodes,
                "features": self.features,
                "labels": self.labels,
            }
            if dataset_str in ["hep_small", "hep_large", "dblp", "cora_enrich"]:
                data["word_adj"] = self.word_adj
                data["word_features"] = self.word_features
            if args.two_stage:
                data["topic_features"] = self.topic_features,
                data["topic_word_adj"] = self.topic_word_adj,
                data["doc_topic_adj"] = self.node_topic_adj,
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_dataset_pickled(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = pickle.load(f)
            self.dataset_str = data["dataset_str"]
            self.adj = data["adj"]
            self.G = data["G"]
            self.G_self_loop = data["G_self_loop"]
            self.node_list = data["node_list"]
            self.train_nodes = data["train_nodes"]
            self.val_nodes = data["val_nodes"]
            self.test_nodes = data["test_nodes"]
            self.features = data["features"]
            self.labels = data["labels"]
            if self.dataset_str in ["hep_small", "hep_large", "dblp","cora_enrich"]:
                self.bert_adj = data["bert_adj"]
                self.w2v_word_adj = data["w2v_word_adj"]
                self.word_adj = data["word_adj"]
                self.word_features = data["word_features"]
                self.bert_features = data["bert_features"]
        print(f"Loaded pickled dataset from {dataset_path}")


class GCNDatasetDeprecated():
    def __init__(self, graph_dataset, mode="train"):
        self.mode = mode
        self.graph_dataset = graph_dataset
        node_list = graph_dataset.node_list
        self.train_masks = [node in graph_dataset.train_nodes for node in node_list]
        self.val_masks = [node in graph_dataset.val_nodes for node in node_list]
        self.test_masks = [node in graph_dataset.test_nodes for node in node_list]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.mode == "train":
            return self.train_masks
        elif self.mode == "validation":
            return self.val_masks
        elif self.mode == "test":
            return self.test_masks
        else:
            print(f"invalid mode {self.mode}")
            raise NotImplementedError


class HeteroGCNDataset(object):
    def __init__(self, graph_dataset, device=None, topic_num=0):
        self.graph_dataset = graph_dataset
        node_list = graph_dataset.node_list
        self.topic_num = topic_num
        self.device = device if device else "cpu"
        self.labels = torch.tensor(graph_dataset.labels).long().to(self.device)
        self.train_masks = [node in graph_dataset.train_nodes for node in node_list]
        self.val_masks = [node in graph_dataset.val_nodes for node in node_list]
        self.test_masks = [node in graph_dataset.test_nodes for node in node_list]
        self.features = graph_dataset.features
        self.word_features = graph_dataset.word_features
        self.num_node = self.features.shape[0]
        self.num_feature = self.word_features.shape[0]
        self.dataset_str = graph_dataset.dataset_str
        self.train_nodes = graph_dataset.train_nodes
        self.val_nodes = graph_dataset.val_nodes
        self.test_nodes = graph_dataset.test_nodes

        # reset adj
        print(len(graph_dataset.adj.nonzero()[0]))
        self.adj = graph_dataset.adj+ sparse.eye(self.num_node)
        self.adj.resize((self.num_node + self.topic_num + self.num_feature, self.num_node + self.topic_num + self.num_feature))

        print('#doc/#topic/#word/: ', self.num_node, self.topic_num, self.num_feature)

        # doc_topic topic_word word_topic
        node_topic_adj = np.zeros(
            (self.num_node + self.topic_num + self.num_feature, self.num_node + self.topic_num + self.num_feature))
        topic_word_adj = np.zeros(
            (self.num_node + self.topic_num + self.num_feature, self.num_node + self.topic_num + self.num_feature))
	
        self.node_graph = dgl.DGLGraph(self.adj)
        self.node_topic_graph = node_topic_adj
        self.topic_node_graph = node_topic_adj.transpose()
        self.topic_word_graph = topic_word_adj
        self.word_topic_graph = topic_word_adj.transpose()

        # node_topic_adj[:self.num_node, self.num_node:self.num_node + self.topic_num] = graph_dataset.node_topic_adj
        # topic_word_adj[self.num_node:self.num_node + self.topic_num, self.num_node + self.topic_num:] = graph_dataset.topic_word_adj
        #
        # self.node_graph = dgl.DGLGraph(self.adj)
        # self.node_topic_graph = dgl.DGLGraph(node_topic_adj)
        # self.topic_node_graph = dgl.DGLGraph(node_topic_adj.transpose())
        # self.topic_word_graph = dgl.DGLGraph(topic_word_adj)
        # self.word_topic_graph = dgl.DGLGraph(topic_word_adj.transpose())

        # features_topic = graph_dataset.topic_features.to(self.device)

        self.ntypes = ["node", "feature", "topic"]
        self.etypes = [("node", "node"), ("node", "topic"), ("topic", "node"), ("topic", "feature"), ("feature", "topic"), ("feature", "feature")]
        if graph_dataset.dataset_str in ["hep_small", "hep_large", "dblp","cora_enrich"]:
            self.word_features = graph_dataset.word_features
            self.word_adj = sparse.csr_matrix(
                np.zeros((self.num_node + self.topic_num + self.num_feature, self.num_node + self.topic_num + self.num_feature)))
            word_adj = graph_dataset.word_adj + sparse.eye(self.num_feature)
            self.word_adj[-self.num_feature:, -self.num_feature:] = word_adj
            # self.word_adj = self.word_adj + sparse.eye(self.num_node + self.topic_num + self.num_feature)
            # self.full_adj = (all_adj + self.word_adj) > 0
            self.feature_graph = dgl.DGLGraph(self.word_adj)
            # self.full_graph = dgl.DGLGraph(self.full_adj)

            feature_feats = torch.tensor(self.word_features)
            # features_topic = torch.tensor(features_topic)

            one_hot_node_feats = torch.tensor(self.features)

            all_adj = self.adj + node_topic_adj + node_topic_adj.transpose() \
                         + topic_word_adj + topic_word_adj.transpose() + self.word_adj

            self.full_graph = all_adj

        else:
            self.feature_graph = dgl.DGLGraph(sparse.eye(self.num_node + self.num_feature + self.topic_num))
            node_feats = torch.tensor(self.features)
            feature_feats = torch.eye(self.num_feature)
            one_hot_node_feats = node_feats
            self.full_graph = dgl.DGLGraph(all_adj)
            features_topic = torch.tensor(features_topic)

        self.features_dict = {
            "one_hot_node": one_hot_node_feats,
            "feature": feature_feats,
            # "topic_feature": features_topic,
        }
        self.graph_dict = {
            ("node", "node"): self.node_graph,
            ("node", "topic"): self.node_topic_graph,
            ("topic", "node"): self.topic_node_graph,
            ("topic", "feature"): self.topic_word_graph,
            ("feature", "topic"): self.word_topic_graph,
            ("feature", "feature"): self.feature_graph,
            "full": self.full_graph,
        }
        self.type2mask = {
            "node": torch.tensor(range(self.num_node)),
            "topic": torch.tensor(range(self.topic_num)) + self.num_node,
            "feature": torch.tensor(range(self.num_feature)) + self.topic_num + self.num_node
        }

    def generate_node_feats(self, pseudo_feature_node_graph, pseudo_node_feature_feats):
        pseudo_feature_node_graph = pseudo_feature_node_graph.local_var()
        pseudo_feature_node_graph.ndata['h'] = pseudo_node_feature_feats
        pseudo_feature_node_graph.update_all(fn.copy_src(src='h', out='m'), fn.mean(msg='m', out='h'))
        return pseudo_feature_node_graph.ndata['h'][:self.num_node]

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sparse.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    def get_data(self):
        return (
            self.ntypes,
            self.etypes,
            self.graph_dict,
            self.features_dict,
            self.type2mask,
            self.train_masks,
            self.val_masks,
            self.test_masks,
            self.labels
        )

    def get_masks(self):
        return self.type2mask, self.train_masks, self.val_masks, self.test_masks, self.labels

    def get_all_graphs(self):
        return [self.graph_dict[etype] for etype in self.etypes]

    def get_graph(self, etype):
        return self.graph_dict[etype]

if __name__ == '__main__':
    # print("Generating ")
    # for dataset_str in ["cora", "citeseer", "pubmed"]:
    #     GraphDataset(dataset_str=dataset_str, root_dir="../data")
    # for dataset_str in ["chameleon", "cornell", "film", "squirrel", "texas", "wisconsin"]:
    #     GraphDataset(dataset_str=dataset_str, root_dir="../data")
    # graph_dataset = GraphDataset(dataset_str="../data/gcn_data/cora.pickle.bin", raw=False)
    # my_dataset = HeteroGCNDataset(graph_dataset=graph_dataset)
    GraphDataset(dataset_str="hep-small", root_dir="../data")
