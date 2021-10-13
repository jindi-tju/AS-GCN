from gensim import corpora
from gensim.models import LdaModel
from gensim import models
from gensim.corpora import Dictionary
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import json
import random
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from scipy import sparse
import copy
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import STOPWORDS

def create_dictionary(msgs):
    dictionary = gensim.corpora.Dictionary(msgs)
    return dictionary

def get_wids(text_doc, bow_dictionary: gensim.corpora.Dictionary):
    row = []
    col = []
    value = []
    row_id = 0
    for d_i, doc in enumerate(text_doc):
        d2b_doc = bow_dictionary.doc2bow(doc)
        for i, j in d2b_doc:
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1
    bow_doc = sparse.coo_matrix((value, (row, col)),
                                shape=(row_id, len(bow_dictionary))).tocsr()
    return bow_doc

def load_doc_data(args):
    root_dir = "data"
    # total_text
    phrased_text = []
    phrased_data = []
    dataset_path = Path(root_dir) / "word_data" / f"{args.dataset_str}"
    with open(dataset_path / args.in_file, 'r') as f:
        for line in f:
            line = line.strip()
            phrased_text.append([line])
            line = line.split(' ')
            phrased_data.append([w for w in line])
    print('输入文本数量：', len(phrased_data))
    # idx_to_word dictionary

    dictionary = create_dictionary(phrased_data)
    bow_title = get_wids(phrased_data, dictionary)

    bow_title_train, bow_title_test = train_test_split(
        bow_title,
        shuffle=True,
        test_size=0.2,
        random_state=42)

    return bow_title_train, bow_title_test, bow_title, dictionary, len(dictionary)
