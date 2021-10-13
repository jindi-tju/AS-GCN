import time
import torch
from src.load_data import load_doc_data
import logging
from torch.optim import Adam
from itertools import chain
from src.model_ntm import NTM
from src.dataset import HeteroGCNDataset, GraphDataset
from src.model import HeteroGATNet, HeteroGCNNet
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import os
import dgl
import json
from scipy import sparse

def time_since(start_time):
    return time.time()-start_time

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
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_topicGCN, optimizer_ntm, optimizer_whole

def train_ntm_one_epoch(model, data_bow, optimizer, args, epoch):
    model.train()
    data_bow = torch.tensor(np.array(data_bow.todense(), dtype=np.float32)).to(args.device)
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
        data_bow = torch.tensor(np.array(data_bow.todense(), dtype=np.float32)).to(args.device)
        data_bow_norm = F.normalize(data_bow)

        _, _, recon_batch, mu, logvar, _ = model(data_bow_norm)
        test_loss = loss_function(recon_batch, data_bow, mu, logvar).item()

    print('====>Test epoch: {} Average loss: {:.4f}'.format(epoch, test_loss))
    logging.info('====> Test epoch: {} Average loss:  {:.4f}'.format(epoch, test_loss))
    return test_loss

def output_theta(model, bow_input):
    theta, _ = model.predict(bow_input)
    print("theta shape", theta.shape)
    # pickle.dump(theta, open(fn, 'wb'))

def output_beta(model, dictionary_bow):
    beta_exp = np.exp(model.get_weights()[-2])
    beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
    # pickle.dump(beta, open(topicWord_fn, 'wb'))
    # with open(topicWordSample_fn, 'w') as fout:
    for k, beta_k in enumerate(beta):
        topic_words = [dictionary_bow[w_id] for w_id in np.argsort(beta_k)[:-11:-1]]
            # fout.write("%s\n" % ' '.join(topic_words))
    print(topic_words)

def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True

def train_model(topicGCN_model, ntm_model, optimizer_topicGCN, optimizer_ntm, optimizer_whole, bow_dictionary, train_bow_loader, valid_bow_loader, total_bow_loader, args, hetero_gcn_dataset):
    logging.info('======================  Start Training  =========================')

    # if args.only_train_ntm or (args.use_topic_represent and not args.load_pretrain_ntm):
    #     print("\nWarming up ntm for %d epochs" % args.ntm_warm_up_epochs)
    #     for epoch in range(1, args.ntm_warm_up_epochs + 1):
    #         sparsity = train_ntm_one_epoch(ntm_model, train_bow_loader, optimizer_ntm, args, epoch)
    #         val_loss = test_ntm_one_epoch(ntm_model, valid_bow_loader, args, epoch)
    #         if epoch % 200 == 0:
    #             ntm_model.print_topic_words(bow_dictionary, os.path.join(args.model_path, 'topwords_e%d.txt' % epoch))
    #             best_ntm_model_path = os.path.join(args.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntm_model' %
    #                                                (epoch, val_loss, sparsity))
    #             logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
    #             torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))
    # elif args.use_topic_represent:
    #     print("Loading ntm model from %s" % args.check_pt_ntm_model_path)
    #     ntm_model.load_state_dict(torch.load(args.check_pt_ntm_model_path))
    #
    # if args.only_train_ntm:
    #     return

    patience = args.patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    state_dict_early_model = None
    curr_step = 0
    dur = []

    device = torch.device(args.device)
    t0 = time.time()
    ntypes, etypes, graph_dict, features_dict, type2mask, train_mask, val_mask, test_mask, labels = hetero_gcn_dataset.get_data()

    feature_feats = features_dict["feature"].to(args.device)

    train_bow_loader = torch.tensor(np.array(train_bow_loader.todense(), dtype=np.float32)).to(device)
    train_bow_norm = F.normalize(train_bow_loader)
    valid_bow_loader = torch.tensor(np.array(valid_bow_loader.todense(), dtype=np.float32)).to(device)
    val_bow_norm = F.normalize(valid_bow_loader)

    total_bow_loader = torch.tensor(np.array(total_bow_loader.todense(), dtype=np.float32)).to(args.device)
    src_bow_norm = F.normalize(total_bow_loader)
    one_hot_node_feats = total_bow_loader

    if args.topic_type == 'z':
        topic_embedding, _ = ntm_model.print_topic_words(bow_dictionary, os.path.join(args.model_path,
                                                                                   'topwords_last.txt'))

        topic_embedding = torch.tensor(topic_representation, requires_grad=False).to(args.device)
        topic_feats = topic_embedding.to(device)

        # print(one_hot_node_feats.size())
        # print(topic_feats.size())
        # print(feature_feats.size())
        # exit()

        empty_node_feats = torch.zeros(
            (one_hot_node_feats.shape[0] + topic_feats.shape[0], feature_feats.shape[1])).to(device)
        empty_topic_feats_1 = torch.zeros((one_hot_node_feats.shape[0], topic_feats.shape[1])).to(
            device)
        empty_topic_feats_2 = torch.zeros((feature_feats.shape[0], topic_feats.shape[1])).to(device)
        empty_feature_feats = torch.zeros(
            (feature_feats.shape[0] + topic_feats.shape[0], one_hot_node_feats.shape[1])).to(device)

        node_dim_feats = torch.cat([one_hot_node_feats.float(), empty_feature_feats])
        topic_dim_feats_1 = torch.cat([empty_topic_feats_1, topic_feats])
        topic_dim_feats = torch.cat([topic_dim_feats_1, empty_topic_feats_2])

        feature_dim_feats = torch.cat([empty_node_feats, feature_feats.float()])
        features = torch.cat((node_dim_feats, topic_dim_feats, feature_dim_feats), dim=1).to(device)

    full_adj = graph_dict["full"]
    node_topic = np.zeros((one_hot_node_feats.shape[0] + args.topic_num + feature_feats.shape[0],
                               one_hot_node_feats.shape[0] + args.topic_num + feature_feats.shape[0]))
    topic_word = np.zeros((one_hot_node_feats.shape[0] + args.topic_num + feature_feats.shape[0],
                               one_hot_node_feats.shape[0] + args.topic_num + feature_feats.shape[0]))
    print("\nEntering main training for %d epochs" % args.epochs)

    for epoch in range(args.start_epoch, args.epochs + 1):
        print("train join")
        logging.info("\nTraining topicgcn+ntm epoch: {}/{}".format(epoch, args.epochs))

        _, topic_word_adj = ntm_model.print_topic_words(bow_dictionary, os.path.join(args.model_path,
                                                                                     'topwords_last.txt'))
        _, _, _, _, _, theta = ntm_model(src_bow_norm)
        # construct doc_topic edges
        doc_topic_old = theta.cpu()
        doc_topic_dis = doc_topic_old.detach().numpy()

        node_topic_adj = np.zeros((src_bow_norm.shape[0], args.topic_num), dtype=np.float32)
        for k, beta_k in enumerate(doc_topic_dis):
            for w_id in np.argsort(beta_k)[:-1 - 1:-1]:
                node_topic_adj[k][w_id] = 1

        node_topic[:one_hot_node_feats.shape[0],
        one_hot_node_feats.shape[0]:one_hot_node_feats.shape[0] + args.topic_num] = node_topic_adj
        topic_word[one_hot_node_feats.shape[0]:one_hot_node_feats.shape[0] + args.topic_num,
        one_hot_node_feats.shape[0] + args.topic_num:] = np.array(topic_word_adj)

        graph_dict[("node", "topic")] = dgl.DGLGraph(node_topic)
        graph_dict[("topic", "feature")] = dgl.DGLGraph(topic_word)
        graph_dict[("feature", "topic")] = dgl.DGLGraph(topic_word.transpose())
        graph_dict[("topic", "node")] = dgl.DGLGraph(node_topic.transpose())
        full_adj_new = full_adj + node_topic + topic_word + node_topic.transpose() + topic_word.transpose()
        graph_dict["full"] = dgl.DGLGraph(full_adj_new)

        g = [graph_dict[etype] for etype in etypes]
        if epoch == 0:
            print(g)

        optimizer = optimizer_topicGCN
        # unfix_model(topicGCN_model)
        topicGCN_model.train()
        # ntm_model.train()

        train_logits = topicGCN_model(g, features)
        # print(train_logits)
        # _, _, recon_batch_train, mu_train, logvar_train, _ = ntm_model(train_bow_norm)
        train_logp = F.log_softmax(train_logits[type2mask["node"]], 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()
        # ntm_loss = loss_function(recon_batch_train, train_bow_norm, mu, logvar)
        train_loss_new = train_loss 
                         # + ntm_loss

        # ntm_loss.backward()
        # train_loss.backward()
        optimizer.zero_grad()
        # train_loss_new.backward()
        train_loss_new.backward()
        optimizer.step()

        topicGCN_model.eval()
        ntm_model.eval()
        with torch.no_grad():
            val_logits = topicGCN_model(g, features)
            _, _, recon_batch_val, mu_val, logvar_val, _ = ntm_model(val_bow_norm)
            val_logp = F.log_softmax(val_logits[type2mask["node"]], 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()
            ntm_loss = loss_function(recon_batch_val, val_bow_norm, mu, logvar)
            val_loss += ntm_loss
            #learning_rate_scheduler.step(val_loss)

            dur.append(time.time() - t0)
            if epoch % args.log_interval == 0 and args.verbose:
                print(
                    "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                        epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))

            if val_acc >= vacc_mx or val_loss <= vlss_mn:
                if val_acc >= vacc_mx and val_loss <= vlss_mn:
                    state_dict_early_model = topicGCN_model.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                vlss_mn = np.min((val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1

        if curr_step >= patience:
            break

    topicGCN_model.load_state_dict(state_dict_early_model)
    # net = load_model['model']
    topicGCN_model.eval()
    ntm_model.eval()
    with torch.no_grad():
        test_logits = topicGCN_model(g, features)
        test_logp = F.log_softmax(test_logits[type2mask["node"]], 1)
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

    # with open(os.path.join('logs', f'{args.model}_results.json'), 'w') as outfile:
    #     outfile.write(json.dumps(results_dict, indent=2) + '\n')


def main(args):
    try:
        start_time = time.time()
        device = torch.device(args.device)
        train_bow, valid_bow, total_bow, bow_dictionary, bow_vocab_size = load_doc_data(args)
        graph_dataset = GraphDataset(args=args, dataset_str=args.dataset_str, bow_dictionary=bow_dictionary, raw=True)
        hetero_gcn_dataset = HeteroGCNDataset(graph_dataset=graph_dataset, device=device, topic_num=args.topic_num)
        ntypes, etypes, graph_dict, features_dict, type2mask, train_mask, val_mask, test_mask, labels = hetero_gcn_dataset.get_data()

        feature_feats = features_dict["feature"].to(device)
        full_graph = dgl.DGLGraph(graph_dict["full"])

        if args.node_feature == "one_hot":
            one_hot_node_feats = torch.tensor(features_dict["one_hot_node"]).to(device)

            node_dim = one_hot_node_feats.shape[1]
            topic_dim = feature_feats.shape[0]
            feature_dim = feature_feats.shape[1]
            in_dim = node_dim + feature_dim + topic_dim
            print("success connect")
        else:
            raise NotImplementedError

        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        start_time = time.time()
        topicGCN_model = HeteroGATNet(g=full_graph, num_input_features=in_dim, num_output_classes=args.out_dim,
                       num_hidden=args.hidden_dim, type2mask=type2mask,
                       num_heads_layer_one=args.num_head, num_heads_layer_two=args.num_head, residual=args.residual,
                       dropout_rate=args.dropout_rate, num_layers=args.num_layer).to(device)

        ntm_model = NTM(args, bow_vocab_size).to(args.device)
        optimizer_topicGCN, optimizer_ntm, optimizer_whole = init_optimizers(topicGCN_model, ntm_model, args)

        train_model(topicGCN_model,
                  ntm_model,
                  optimizer_topicGCN,
                  optimizer_ntm,
                  optimizer_whole,
                  bow_dictionary,
                  train_bow,
                  valid_bow,
                  total_bow,
                  args,
                  hetero_gcn_dataset)

        training_time = time_since(start_time)

        logging.info('Time for training: %.1f' % training_time)

    except Exception as e:
        logging.exception("message")
    return