"""
This file is a simple templet for final version of the code. Transfer to pytorch-lightning for publication
"""


import dgl
import pytorch_lightning as ptl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataset import GCNDataset, GraphDataset


def my_follate_fn(batch_data):
    return batch_data[0]


class GCN(ptl.LightningModule):
    def __init__(self, hparams):
        super(GCN, self).__init__()
        self.hparams = hparams
        self.dataset = GraphDataset(dataset_str=hparams.dataset_str, raw=False)
        self.features = torch.tensor(self.dataset.features).float()
        self.labels = torch.tensor(self.dataset.labels)
        self.g = dgl.DGLGraph(self.dataset.G_self_loop)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(hparams.in_feats, hparams.n_hidden, activation=hparams.activation))

        for i in range(hparams.n_layers - 1):
            self.layers.append(GraphConv(hparams.n_hidden, hparams.n_hidden, activation=hparams.activation))
        self.layers.append(GraphConv(hparams.n_hidden, hparams.out_feats))
        self.dropout = nn.Dropout(p=hparams.dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        logits = self.forward(self.features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[batch], self.labels[batch])
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        logits = self.forward(self.features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[batch], self.labels[batch])
        labels = self.labels[batch]
        _, indices = torch.max(logp[batch], dim=1)
        correct = torch.sum(indices == labels)
        return {'val_loss': loss, 'correct': correct, 'labels': torch.tensor(labels.shape[0])}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        labels = torch.stack([x['labels'] for x in outputs]).sum().float()
        return {'val_loss': val_loss_mean, 'acc': correct / labels}

    def test_step(self, batch, batch_nb):
        logits = self.forward(self.features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[batch], self.labels[batch])
        labels = self.labels[batch]
        _, indices = torch.max(logp[batch], dim=1)
        correct = torch.sum(indices == labels)
        return {'test_loss': loss, 'correct': correct, 'labels': torch.tensor(labels.shape[0])}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        labels = torch.stack([x['labels'] for x in outputs]).sum().float()
        return {'test_loss': test_loss_mean, 'acc': correct / labels}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
        return [optimizer], [scheduler]

    @ptl.data_loader
    def train_dataloader(self):
        return DataLoader(dataset=GCNDataset(graph_dataset=self.dataset, mode="train"), collate_fn=my_follate_fn,
                          num_workers=self.hparams.num_workers)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(dataset=GCNDataset(graph_dataset=self.dataset, mode="validation"), collate_fn=my_follate_fn,
                          num_workers=self.hparams.num_workers)

    @ptl.data_loader
    def test_dataloader(self):
        return DataLoader(dataset=GCNDataset(graph_dataset=self.dataset, mode="test"), collate_fn=my_follate_fn,
                          num_workers=self.hparams.num_workers)
