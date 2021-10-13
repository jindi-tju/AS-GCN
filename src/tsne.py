# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import re


def load_data():
    data_set = 'DBLP'
    path = r'data\preprocessed\{}_processed'.format(data_set)

    label = np.load(os.path.join(path, 'labels.npy'))
    idx = np.load(os.path.join(path, 'train_val_test_idx.npz'))
    idx_test = idx['test_idx']

    with open('{}-emb.txt'.format(data_set), 'r') as f:
        line = f.readline()
        line = line.strip()
        line = re.split(' ', line)
        emb = np.zeros((int(line[0]), int(line[1])), dtype=np.float)
        for i in range(emb.shape[0]):
            line = f.readline()
            line = line.strip()
            line = re.split(' ', line)
            id = int(line[0])
            emb[id] = line[1:]

    return emb, label[idx_test]


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_tsne(data, label, dataset_str):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    t1 = time()
    fig = plt.figure(figsize=(8, 8))
    from matplotlib.ticker import NullFormatter
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(result[:, 0], result[:, 1], c=label, cmap=plt.cm.Spectral)
    plt.title("t-SNE")
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.show()

    # plt.savefig('/media/user/2FD84FB396A78049/Yuzz/BIGCN_topic/plot_result/plot_tsne.png')
    plt.savefig('/media/user/2FD84FB396A78049/Yuzz/BIGCN_topic_0319/plot_result/plot_tsne_TopicGCN_{}.png'.format(dataset_str))

