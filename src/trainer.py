# node_node, node_feature, feature_node, feature_feature
import json
import os
import time

import dgl
import numpy as np
import torch
from scipy import sparse
from torch.functional import F
import torch.nn as nn
from src.dataset import HeteroGCNDataset, GraphDataset
from src.model import HeteroGATNet, HeteroGCNNet
from src.load_data import load_doc_data
from src.model_old import HeteroGCN, VanillaGCN, VanillaGAT, HeteroMultiHeadGCN
from torch.optim import Adam
from itertools import chain
from src.model_ntm import NTM
import logging
from sklearn.metrics import f1_score, accuracy_score
from src.tsne import plot_tsne

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)

    def train(self):
        for epoch in range(self.args.epochs):
            t0 = time.time()
            self.net.train()
            train_logits = self.net(self.g, self.features)
            train_logp = F.log_softmax(train_logits, 1)
            train_loss = F.nll_loss(train_logp[self.train_mask], self.labels[self.train_mask])
            train_pred = train_logp.argmax(dim=1)
            train_acc = torch.eq(train_pred[self.train_mask], self.labels[self.train_mask]).float().mean().item()
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(self.g, self.features)
                val_logp = F.log_softmax(val_logits, 1)
                val_loss = F.nll_loss(val_logp[self.val_mask], self.labels[self.val_mask]).item()
                val_pred = val_logp.argmax(dim=1)
                val_acc = torch.eq(val_pred[self.val_mask], self.labels[self.val_mask]).float().mean().item()

            self.learning_rate_scheduler.step(val_loss)

            self.dur.append(time.time() - t0)
            if epoch % self.args.log_interval == 0 and self.args.verbose:
                print(
                    "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                        epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(self.dur) / len(self.dur)))
            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = self.net.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= self.patience:
                    break
        self.net.load_state_dict(state_dict_early_model)
        self.net.eval()
        with torch.no_grad():
            test_logits = self.net(self.g, self.features)
            test_logp = F.log_softmax(test_logits, 1)
            test_loss = F.nll_loss(test_logp[self.test_mask], self.labels[self.test_mask]).item()
            test_pred = test_logp.argmax(dim=1)
            test_acc = torch.eq(test_pred[self.test_mask], self.labels[self.test_mask]).float().mean().item()

        print("Test_acc" + ":" + str(test_acc))

        results_dict = vars(self.args)
        results_dict['test_loss'] = test_loss
        results_dict['test_acc'] = test_acc
        results_dict['actual_epochs'] = 1 + epoch
        results_dict['val_acc_max'] = vacc_mx
        results_dict['val_loss_min'] = vlss_mn
        results_dict['total_time'] = sum(self.dur)

        with open(os.path.join('logs', f'{self.args.model}_results.json'), 'w') as outfile:
            outfile.write(json.dumps(results_dict, indent=2) + '\n')

class HeteroGCNTrainer(BaseTrainer):
    def __init__(self, args):
        super(HeteroGCNTrainer, self).__init__(args)
        self.hetero_gcn_dataset = HeteroGCNDataset(graph_dataset=self.graph_dataset, device=self.device)
        self.ntypes, self.etypes, self.graph_dict, self.features_dict, self.type2mask, self.train_mask, self.val_mask, self.test_mask, self.labels = self.hetero_gcn_dataset.get_data()
        self.g = [self.graph_dict[etype] for etype in self.etypes]
        if args.word_emb == "w2v":
            self.feature_feats = self.features_dict["word_w2v"]
        elif args.word_emb == "jose":
            self.feature_feats = self.features_dict["word_jose"]
        else:
            raise NotImplementedError
        if args.node_feature == "message_passing":
            self.node_feats = self.features_dict["node"]
            self.features = torch.cat([self.node_feats, self.feature_feats]).to(self.device)
            self.in_dim = self.features.shape[1]
            self.net = HeteroGCN(in_dim=self.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                                 num_layers=args.num_layer,
                                 ntypes=self.ntypes,
                                 etypes=self.etypes,
                                 type2mask=self.type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                                 in_dropout=args.dropout_rate,
                                 hidden_dropout=args.dropout_rate, output_dropout=0.1).to(self.device)
        elif args.node_feature == "one_hot":
            self.one_hot_node_feats = self.features_dict["one_hot_node"]
            self.empty_node_feats = torch.zeros((self.one_hot_node_feats.shape[0], self.feature_feats.shape[1]))
            self.empty_feature_feats = torch.zeros((self.feature_feats.shape[0], self.one_hot_node_feats.shape[1]))
            self.node_dim_feats = torch.cat([self.one_hot_node_feats.float(), self.empty_feature_feats]).to(self.device)
            self.node_dim = self.one_hot_node_feats.shape[1]
            self.feature_dim_feats = torch.cat([self.empty_node_feats, self.feature_feats.float()]).to(self.device)
            self.feature_dim = self.feature_feats.shape[1]
            self.features = [self.node_dim_feats, self.node_dim_feats, self.feature_dim_feats, self.feature_dim_feats]
            self.in_dims = [self.node_dim, self.node_dim, self.feature_dim, self.feature_dim]
            self.hidden_dims = args.hidden_dim
            self.out_dims = args.out_dim
            self.net = HeteroGCN(in_dim=self.in_dims, hidden_dim=self.hidden_dims, out_dim=self.out_dims,
                                 num_layers=args.num_layer,
                                 ntypes=self.ntypes,
                                 etypes=self.etypes,
                                 type2mask=self.type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                                 in_dropout=args.dropout_rate,
                                 hidden_dropout=args.dropout_rate, output_dropout=0.1).to(self.device)
        elif args.node_feature == "bert":
            self.bert_feats = self.features_dict["node_bert"]
            self.empty_node_feats = torch.zeros((self.bert_feats.shape[0], self.feature_feats.shape[1]))
            self.empty_feature_feats = torch.zeros((self.feature_feats.shape[0], self.bert_feats.shape[1]))
            self.node_dim_feats = torch.cat([self.bert_feats.float(), self.empty_feature_feats]).to(self.device)
            self.node_dim = self.bert_feats.shape[1]
            self.feature_dim_feats = torch.cat([self.empty_node_feats, self.feature_feats.float()]).to(self.device)
            self.feature_dim = self.feature_feats.shape[1]
            self.features = [self.node_dim_feats, self.node_dim_feats, self.feature_dim_feats, self.feature_dim_feats]
            self.in_dims = [self.node_dim, self.node_dim, self.feature_dim, self.feature_dim]
            self.net = HeteroGCN(in_dim=self.in_dims, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                                 num_layers=args.num_layer,
                                 ntypes=self.ntypes,
                                 etypes=self.etypes,
                                 type2mask=self.type2mask, activation=F.leaky_relu, aggr_func=args.aggr_func,
                                 in_dropout=args.dropout_rate,
                                 hidden_dropout=args.dropout_rate, output_dropout=0.1).to(self.device)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                                 factor=args.learning_rate_decay_factor,
                                                                                  patience=args.learning_rate_decay_patience,
                                                                                  verbose=args.verbose)
        self.patience = args.patience
        self.vlss_mn = np.inf
        self.vacc_mx = 0.0
        self.state_dict_early_model = None
        self.curr_step = 0
        self.dur = []

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))

def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def init_optimizers(topicGCN_model, ntm_model, opt):
    optimizer_topicGCN = Adam(params=filter(lambda p: p.requires_grad, topicGCN_model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate_ntm)
    whole_params = chain(topicGCN_model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    return optimizer_topicGCN, optimizer_ntm, optimizer_whole

def train_ntm_one_epoch(model, data_bow, optimizer, args, epoch):
    model.train()
    # data_bow = torch.tensor(np.array(data_bow.todense(), dtype=np.float32)).to(args.device)
    # normalize data
    data_bow_norm = F.normalize(data_bow)
    optimizer.zero_grad()
    _, _, recon_batch, mu, logvar, _ = model(data_bow_norm)
    loss = loss_function(recon_batch, data_bow, mu, logvar)
    loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
    loss.backward()
    train_loss = loss.item()
    optimizer.step()

    print('====>Train epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    logging.info('====>Train epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    sparsity = check_sparsity(model.fcd1.weight.data)
    logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    logging.info("Target sparsity = %.3f" % args.target_sparsity)
    update_l1(model.l1_strength, sparsity, args.target_sparsity)
    return sparsity

def test_ntm_one_epoch(model, data_bow, args, epoch):
    model.eval()
    # test_loss = 0
    with torch.no_grad():
        # data_bow = torch.tensor(np.array(data_bow.todense(), dtype=np.float32)).to(args.device)
        data_bow_norm = F.normalize(data_bow)

        _, _, recon_batch, mu, logvar, _ = model(data_bow_norm)
        test_loss = loss_function(recon_batch, data_bow, mu, logvar).item()

    print('====>Test epoch: {} Average loss: {:.4f}'.format(epoch, test_loss))
    logging.info('====> Test epoch: {} Average loss:  {:.4f}'.format(epoch, test_loss))
    return test_loss

def init_optimizers(topicGCN_model, ntm_model, opt):
    optimizer_topicGCN = Adam(params=filter(lambda p: p.requires_grad, topicGCN_model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate_ntm)
    whole_params = chain(topicGCN_model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_topicGCN, optimizer_ntm, optimizer_whole

def hetero_gat(args):
    device = torch.device(args.device)
    train_bow, valid_bow, total_bow, bow_dictionary, bow_vocab_size = load_doc_data(args)

    train_bow_loader = torch.tensor(np.array(train_bow.todense(), dtype=np.float32)).to(device)
    train_bow_norm = F.normalize(train_bow_loader)
    valid_bow_loader = torch.tensor(np.array(valid_bow.todense(), dtype=np.float32)).to(device)
    val_bow_norm = F.normalize(valid_bow_loader)
    total_bow_loader = torch.tensor(np.array(total_bow.todense(), dtype=np.float32)).to(args.device)
    src_bow_norm = F.normalize(total_bow_loader)

    graph_dataset = GraphDataset(args=args, dataset_str=args.dataset_str, bow_dictionary=bow_dictionary, raw=True)
    hetero_gcn_dataset = HeteroGCNDataset(graph_dataset=graph_dataset, device=device, topic_num=args.topic_num)
    ntypes, etypes, graph_dict, features_dict, type2mask, train_mask, val_mask, test_mask, labels = hetero_gcn_dataset.get_data()

    feature_feats = features_dict["feature"].to(device)
    # one_hot_node_feats = torch.tensor(features_dict["one_hot_node"]).to(device)
    full_adj = graph_dict["full"]
    full_graph = dgl.DGLGraph(sparse.csr_matrix(graph_dict["full"]))
    # graph_dict["full"] = dgl.DGLGraph(full_adj_new)

    node_dim = feature_feats.shape[0]
    topic_dim = feature_feats.shape[0]
    feature_dim = feature_feats.shape[1]
    in_dim = node_dim + feature_dim + topic_dim
    print("结点维度" + "\t" + str(node_dim))
    print("特征维度" + "\t" + str(feature_dim))
    print("主题维度" + "\t" + str(topic_dim))

    net = HeteroGATNet(g=full_graph, num_input_features=in_dim, num_output_classes=args.out_dim,
                       num_hidden=args.hidden_dim, type2mask=type2mask,
                       num_heads_layer_one=args.num_head, num_heads_layer_two=args.num_head, residual=args.residual,
                       dropout_rate=args.dropout_rate, num_layers=args.num_layer).to(device)
    ntm_model = NTM(args, bow_vocab_size).to(args.device)
    optimizer_topicGCN, optimizer_ntm, optimizer_whole = init_optimizers(net, ntm_model, args)

    min_bound_ntm = np.inf
    curr_step_ntm = 0
    patience = args.patience
    state_dict_early_ntm_model = None
    if args.only_train_ntm or (args.use_topic_represent and not args.load_pretrain_ntm):
        print("\nWarming up ntm for %d epochs" % args.ntm_warm_up_epochs)
        for epoch in range(1, args.ntm_warm_up_epochs + 1):
            sparsity = train_ntm_one_epoch(ntm_model, total_bow_loader[train_mask], optimizer_ntm, args, epoch)
            val_loss = test_ntm_one_epoch(ntm_model, total_bow_loader[val_mask], args, epoch)
            if val_loss < min_bound_ntm:
                min_bound_ntm = val_loss
                print("Saving model")
                state_dict_early_ntm_model = ntm_model.state_dict()
                curr_step_ntm = 0
            else:
                curr_step_ntm += 1
                if curr_step_ntm >= patience:
                    break
            # if epoch % 200 == 0:
            #     ntm_model.print_topic_words(bow_dictionary, os.path.join(args.model_path, 'topwords_e%d.txt' % epoch))
            #     best_ntm_model_path = os.path.join(args.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntm_model' %
            #                                        (epoch, val_loss, sparsity))
            #     logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
            #     torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))
    elif args.use_topic_represent:
        print("Loading ntm model from %s" % args.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(args.check_pt_ntm_model_path))

    if args.only_train_ntm:
        return

    ntm_model.load_state_dict(state_dict_early_ntm_model)

    node_topic = np.zeros((total_bow_loader.shape[0] + args.topic_num + feature_feats.shape[0],
                           total_bow_loader.shape[0] + args.topic_num + feature_feats.shape[0]))
    topic_word = np.zeros((total_bow_loader.shape[0] + args.topic_num + feature_feats.shape[0],
                           total_bow_loader.shape[0] + args.topic_num + feature_feats.shape[0]))

    # if args.node_feature == "one_hot":
    one_hot_node_feats = total_bow_loader
    rep1 = nn.Linear(one_hot_node_feats.shape[1], one_hot_node_feats.shape[1]).to(args.device)
    # relu = nn.ReLU()
    one_hot_node_feats = torch.tensor(rep1(one_hot_node_feats), requires_grad=False).to(args.device)

    empty_node_feats = torch.zeros((one_hot_node_feats.shape[0] + args.topic_num, feature_feats.shape[1])).to(device)
    empty_topic_feats_1 = torch.zeros((one_hot_node_feats.shape[0], one_hot_node_feats.shape[1])).to(device)
    empty_topic_feats_2 = torch.zeros((feature_feats.shape[0], one_hot_node_feats.shape[1])).to(device)
    empty_feature_feats = torch.zeros((feature_feats.shape[0] + args.topic_num, one_hot_node_feats.shape[1])).to(device)
    node_dim_feats = torch.cat([one_hot_node_feats.float(), empty_feature_feats])
    feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()])

    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    optimizer = optimizer_whole
    for epoch in range(args.epochs):
        # print(epoch)
        t0 = time.time()

        topic_representation, topic_word_adj = ntm_model.print_topic_words(bow_dictionary, epoch, os.path.join(args.model_path,
                                                                                                        'topwords_last.txt'), args.topwords)
        topic_feats = torch.tensor(topic_representation, requires_grad=False).to(args.device)
        topic_dim_feats_1 = torch.cat([empty_topic_feats_1, topic_feats])
        topic_dim_feats = torch.cat([topic_dim_feats_1, empty_topic_feats_2])
        features = torch.cat((node_dim_feats, topic_dim_feats, feature_dim_feats), dim=1).to(device)

        _, _, _, _, _, theta = ntm_model(src_bow_norm)
        # construct doc_topic edges
        doc_topic_old = theta.cpu()
        doc_topic_dis = doc_topic_old.detach().numpy()

        toptopic = args.toptopic
        node_topic_adj = np.zeros((src_bow_norm.shape[0], args.topic_num), dtype=np.float32)
        for k, beta_k in enumerate(doc_topic_dis):
            for w_id in np.argsort(beta_k)[:-toptopic - 1:-1]:
                node_topic_adj[k][w_id] = 1

        node_topic[:one_hot_node_feats.shape[0],
        one_hot_node_feats.shape[0]:one_hot_node_feats.shape[0] + args.topic_num] = node_topic_adj
        topic_word[one_hot_node_feats.shape[0]:one_hot_node_feats.shape[0] + args.topic_num,
        one_hot_node_feats.shape[0] + args.topic_num:] = np.array(topic_word_adj)

        graph_dict[("node", "topic")] = dgl.DGLGraph(sparse.csr_matrix(node_topic))
        graph_dict[("topic", "feature")] = dgl.DGLGraph(sparse.csr_matrix(topic_word))
        graph_dict[("feature", "topic")] = dgl.DGLGraph(sparse.csr_matrix(topic_word.transpose()))
        graph_dict[("topic", "node")] = dgl.DGLGraph(sparse.csr_matrix(node_topic.transpose()))
        full_adj_new = full_adj + node_topic + topic_word + node_topic.transpose() + topic_word.transpose()
        graph_dict["full"] = dgl.DGLGraph(sparse.csr_matrix(full_adj_new))

        g = [graph_dict[etype] for etype in etypes]

        if epoch == 0:
            print(g)

        net.train()
        ntm_model.train()
        # print(features.shape)
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits[type2mask["node"]], 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()
        # y_true_label = np.argmax(label_test, axis=1)
        # train_acc_new = accuracy_score(train_pred[train_mask].cpu(), labels[train_mask].cpu())
        train_f1 = f1_score(train_pred[train_mask].cpu(), labels[train_mask].cpu(), average="weighted")
        _, _, recon_batch_train, mu_train, logvar_train, _ = ntm_model(src_bow_norm[train_mask])
        ntm_loss = loss_function(recon_batch_train, src_bow_norm[train_mask], mu_train, logvar_train)
        train_loss += args.lamuda * ntm_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        ntm_model.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits[type2mask["node"]], 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()
            val_f1 = f1_score(val_pred[val_mask].cpu(), labels[val_mask].cpu(), average="weighted")
            _, _, recon_batch_val, mu_val, logvar_val, _ = ntm_model(src_bow_norm[val_mask])
            ntm_loss = loss_function(recon_batch_val, src_bow_norm[val_mask], mu_val, logvar_val)
            val_loss += args.lamuda * ntm_loss

        # learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Train F1 {:.4f} | Val Loss {:.4f} | Val F1 {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss, train_acc, train_f1, val_loss, val_f1, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
                state_dict_early_ntm_model = ntm_model.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break

    net.eval()
    ntm_model.eval()
    with torch.no_grad():
        net.load_state_dict(state_dict_early_model)
        ntm_model.load_state_dict(state_dict_early_ntm_model)

        topic_representation, topic_word_adj = ntm_model.print_topic_words(bow_dictionary, 0, os.path.join(args.model_path,
                                                                                     'topwords_last.txt'), args.topwords)
        _, _, _, _, _, theta = ntm_model(src_bow_norm)
        # construct doc_topic edges
        doc_topic_old = theta.cpu()
        doc_topic_dis = doc_topic_old.detach().numpy()

        node_topic_adj = np.zeros((src_bow_norm.shape[0], args.topic_num), dtype=np.float32)
        for k, beta_k in enumerate(doc_topic_dis):
            for w_id in np.argsort(beta_k)[:toptopic - 1:-1]:
                node_topic_adj[k][w_id] = 1

        node_topic[:one_hot_node_feats.shape[0],
        one_hot_node_feats.shape[0]:one_hot_node_feats.shape[0] + args.topic_num] = node_topic_adj
        topic_word[one_hot_node_feats.shape[0]:one_hot_node_feats.shape[0] + args.topic_num,
        one_hot_node_feats.shape[0] + args.topic_num:] = np.array(topic_word_adj)

        graph_dict[("node", "topic")] = dgl.DGLGraph(sparse.csr_matrix(node_topic))
        graph_dict[("topic", "feature")] = dgl.DGLGraph(sparse.csr_matrix(topic_word))
        graph_dict[("feature", "topic")] = dgl.DGLGraph(sparse.csr_matrix(topic_word.transpose()))
        graph_dict[("topic", "node")] = dgl.DGLGraph(sparse.csr_matrix(node_topic.transpose()))
        full_adj_new = full_adj + node_topic + topic_word + node_topic.transpose() + topic_word.transpose()
        graph_dict["full"] = dgl.DGLGraph(sparse.csr_matrix(full_adj_new))

        g = [graph_dict[etype] for etype in etypes]

        topic_feats = torch.tensor(topic_representation, requires_grad=False).to(args.device)
        topic_dim_feats_1 = torch.cat([empty_topic_feats_1, topic_feats])
        topic_dim_feats = torch.cat([topic_dim_feats_1, empty_topic_feats_2])
        features = torch.cat((node_dim_feats, topic_dim_feats, feature_dim_feats), dim=1).to(device)

        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits[type2mask["node"]], 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()
        test_f1 = f1_score(test_pred[test_mask].cpu(), labels[test_mask].cpu(), average="weighted")
        macro_test_f1 = f1_score(test_pred[test_mask].cpu(), labels[test_mask].cpu(), average="macro")
        # if args.lamuda == 1.0:
            # plot_tsne(test_logp.cpu(), labels.cpu(), args.dataset_str)
    print("Test_acc" + ":" + str(test_acc))
    print("Test_f1" + ":" + str(test_f1))
    print("Macro_Test_f1" + ":" + str(macro_test_f1))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_f1'] = test_f1
    results_dict['macro_test_f1'] = macro_test_f1
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_{args.dataset_str}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')


def vanilla_gcn(args):
    device = torch.device(args.device)
    graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)

    train_mask = [node in graph_dataset.train_nodes for node in graph_dataset.node_list]
    val_mask = [node in graph_dataset.val_nodes for node in graph_dataset.node_list]
    test_mask = [node in graph_dataset.test_nodes for node in graph_dataset.node_list]
    labels = torch.tensor(graph_dataset.labels).to(device)
    g = dgl.DGLGraph(graph_dataset.G_self_loop)
    features = torch.tensor(graph_dataset.features).float().to(device)
    in_dim = features.shape[1]
    net = VanillaGCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_layers=args.num_layer,
                     activation=F.leaky_relu, in_dropout=0.1,
                     hidden_dropout=args.dropout_rate,
                     output_dropout=0.1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                         factor=args.learning_rate_decay_factor,
                                                                         patience=args.learning_rate_decay_patience,
                                                                         verbose=args.verbose)
    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()

        net.train()
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        # learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    net.load_state_dict(state_dict_early_model)
    net.eval()
    with torch.no_grad():
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    print("Test_acc" + ":" + str(test_acc))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')


def vanilla_gat(args):
    device = torch.device(args.device)
    graph_dataset = GraphDataset(dataset_str=args.dataset, raw=False)

    train_mask = [node in graph_dataset.train_nodes for node in graph_dataset.node_list]
    val_mask = [node in graph_dataset.val_nodes for node in graph_dataset.node_list]
    test_mask = [node in graph_dataset.test_nodes for node in graph_dataset.node_list]
    labels = torch.tensor(graph_dataset.labels).to(device)
    g = dgl.DGLGraph(graph_dataset.G_self_loop)
    features = torch.tensor(graph_dataset.features).float().to(device)
    in_dim = features.shape[1]
    net = VanillaGAT(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, num_layers=args.num_layer,
                     activation=F.leaky_relu, residual=args.residual, feat_drop=args.dropout_rate,
                     heads=[args.num_head] * (1 + args.num_layer), attn_drop=args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                         factor=args.learning_rate_decay_factor,
                                                                         patience=args.learning_rate_decay_patience,
                                                                         verbose=args.verbose)
    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()

        net.train()
        train_logits = net(g, features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        # learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)
        if epoch % args.log_interval == 0 and args.verbose:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
    net.load_state_dict(state_dict_early_model)
    net.eval()
    with torch.no_grad():
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()

    print("Test_acc" + ":" + str(test_acc))

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)

    with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
        outfile.write(json.dumps(results_dict, indent=2) + '\n')
