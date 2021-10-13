import torch.nn as nn
import torch
from torch.nn import functional as F
import logging
import numpy as np

class NTM(nn.Module):
    def __init__(self, opt, bow_vocab_size, hidden_dim=500, l1_strength=0.001):
        super(NTM, self).__init__()
        self.input_dim = bow_vocab_size
        self.topic_num = opt.topic_num
        topic_num = opt.topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(opt.device)
        self.rep1 = nn.Linear(self.input_dim, self.input_dim)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def topic_represent(self, w):
        return F.relu(self.rep1(w))

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        # doc_topic dis
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)
        # print(theta.shape)
        # print(theta.cpu().detach().numpy())
        # print(np.amax(theta.cpu().detach().numpy(), axis=1))
        # exit()
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar, theta

    def print_topic_words(self, vocab_dic, epoch, fn, n_top_words=10):
        topic_representation = self.topic_represent(self.fcd1.weight.data.T)
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        topic_word_adj = np.zeros((self.topic_num, beta_exp.shape[1]), dtype=np.float32)
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = []
            for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]:
                topic_word_adj[k][w_id] = 1
                topic_words.append(vocab_dic[w_id])
            if epoch == 0:
                print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()
        return topic_representation, topic_word_adj