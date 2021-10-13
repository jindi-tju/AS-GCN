import argparse
from torch.optim import Adam
import torch
from torch.nn import functional as F
import torch.nn as nn
import logging
import os
from src.load_data import load_doc_data
import numpy as np
from src.model_ntm import NTM

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

def init_optimizers(model, ntm_model, opt):
    optimizer_seq2seq = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_seq2seq, optimizer_ntm, optimizer_whole

def train_ntm_one_epoch(model, data_bow, optimizer, args, epoch):
    model.train()
    train_loss = 0

    data_bow = data_bow.to(args.device)
    # data_bow = torch.tensor(data_bow,).to(args.device)
    # normalize data
    data_bow_norm = F.normalize(data_bow)
    optimizer.zero_grad()
    _, _, recon_batch, mu, logvar, _ = model(data_bow_norm)
    loss = loss_function(recon_batch, data_bow, mu, logvar)
    loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
    loss.backward()
    train_loss = loss.item()
    # train_loss += loss.item()
    optimizer.step()

    print('====>Train epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    logging.info('====>Train epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    sparsity = check_sparsity(model.fcd1.weight.data)
    logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    logging.info("Target sparsity = %.3f" % args.target_sparsity)
    update_l1(model.l1_strength, sparsity, args.target_sparsity)
    return sparsity

def test_ntm_one_epoch(model, data_bow, opt, epoch):
    model.eval()
    # test_loss = 0
    with torch.no_grad():
        data_bow = data_bow.to(opt.device)
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

def init_optimizers(model, ntm_model, args):
    optimizer_seq2seq = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=args.learning_rate_ntm)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_seq2seq, optimizer_ntm, optimizer_whole

def NTM_model(args):
    device = torch.device(args.device)
    train_bow, valid_bow, total_bow, bow_dictionary, bow_vocab_size = load_doc_data(args)
    train_bow = torch.tensor(np.array(train_bow.todense(), dtype=np.float32)).to(device)
    valid_bow = torch.tensor(np.array(valid_bow.todense(), dtype=np.float32)).to(device)

    ntm_model = NTM(args, bow_vocab_size).to(args.device)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=args.learning_rate_ntm)

    for epoch in range(1, args.ntm_warm_up_epochs + 1):
        print(epoch)
        sparsity = train_ntm_one_epoch(ntm_model, train_bow, optimizer_ntm, args, epoch)
        val_loss= test_ntm_one_epoch(ntm_model, valid_bow, args, epoch)
        if epoch % 50 == 0:
            # ntm_model.print_doc_topic(os.path.join(args.model_path, 'topwords_e%d.txt' % epoch))
            best_ntm_model_path = os.path.join(args.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntm_model' %
                                               (epoch, val_loss, sparsity))
            logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
            torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))


    print("Loading ntm model from %s" % best_ntm_model_path)
    ntm_model.load_state_dict(torch.load(best_ntm_model_path))
    total_bow = torch.tensor(np.array(total_bow.todense(), dtype=np.float32)).to(device)
    total_bow_norm = F.normalize(total_bow)

    # topic_words
    topic_embedding, topic_word_adj = ntm_model.print_topic_words(bow_dictionary, os.path.join(args.model_path, 'topwords_last.txt'))

    if args.topic_type == 'z':
        doc_represent, topic_represent, recon_batch, mu, logvar, theta = ntm_model(total_bow_norm)
        # construct doc_topic edges
        doc_topic_old = theta.cpu()
        doc_topic_dis = doc_topic_old.detach().numpy()

        node_topic_adj = np.zeros((total_bow.shape[0], args.topic_num), dtype=np.float32)
        # only top_one
        # topic_represent_pred = g.argmax(dim=1)
        # topic_represent_1 = topic_represent_pred.cpu()
        # for k, beta_k in enumerate(topic_represent_1):
        #     node_topic_adj[k][beta_k] = 1
        # print(node_topic_adj)
        # exit()
        for k, beta_k in enumerate(doc_topic_dis):
            for w_id in np.argsort(beta_k)[:-1 - 1:-1]:
                node_topic_adj[k][w_id]=1
        #     topic_words = [w_id for w_id in np.argsort(beta_k)[:-2 - 1:-1]]
        # print(topic_words)

        # for k, beta_k in enumerate(doc_topic_1_dis):
        #     topic_words_1 = [w_id for w_id in np.argsort(beta_k)[:-2 - 1:-1]]
        # print(topic_words_1)
        # exit()
    else:
        _, topic_represent, recon_batch, mu, logvar = ntm_model(total_bow_norm)
        # print(topic_represent)
        # ntm_model.print_doc_topic(os.path.join(args.model_path, 'Doc_topic_last.txt'), topic_represent)

    return topic_embedding, topic_word_adj, doc_represent, node_topic_adj, bow_dictionary
