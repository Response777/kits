# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
from gensim.models import KeyedVectors
from sklearn.base import TransformerMixin

from commons import log
log = log(True)

class SemEvalVectorizer(TransformerMixin):
    """
        Transform formatted sentences into indices array to feed into other model.
        It inherits TransformerMixin from sklearn to make it work with scikit-learn.
    """
    def __init__(self, config = {}):
        self.transform_position = True
        self.word2vec           = None
        self.word2vec_path      = None
        self.word_embed_dim     = None
        self.vocab              = None
        self.all_labels         = None
        self.padded_length      = 0
        self.load_config(config)
        self.load_word2vec()

    def load_config(self, config):
        for k,v in config.items():
            setattr(self, k, v)

    @log
    def load_word2vec(self):
        print('Loading '+ self.word2vec_path.split(os.sep)[-1])
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
        self.word_embed_dim = self.word2vec.syn0.shape[1]
        # self.word2vec = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)

    @log
    def transform(self, mentions):
        '''
            tokens -> sentence_idx, pos_e1, pos_e2
        '''
        mentions_out = np.zeros([len(mentions), self.padded_length], np.int32)
        labels_out   = np.zeros([len(mentions)], np.int32)
        pf_e1        = np.zeros([len(mentions), self.padded_length], np.int32)
        pf_e2        = np.zeros([len(mentions), self.padded_length], np.int32)

        for i, mention in enumerate(mentions):
            tokens = mention['tokens']
            pos_e1 = mention['pos_e1']
            pos_e2 = mention['pos_e2']
            label  = mention['label']

            labels_out = self.all_labels.index(label)
            tokens = [w.lower() for w in tokens]
            tokens = tokens + ['_EOS'] + ['_PAD'] * (self.padded_length - 1 - len(tokens))
            for j, token in enumerate(tokens):
                mentions_out[i][j] = self.vocab.index(token)
                pf_e1[i][j]        = j - pos_e1 + self.padded_length - 1
                pf_e2[i][j]        = j - pos_e2 + self.padded_length - 1

        return mentions_out, labels_out, pf_e1, pf_e2

    @log
    def fit(self, mentions_all, y = None, **fit_params):
        '''
            decide the size of position_mat, build and sort the vocab.
        '''
        self.max_length = 0
        vocab_count     = {}
        for mention in mentions_all:
            tokens = mention['tokens']
            self.max_length = max([len(tokens),self.max_length])
            for token in tokens:
                token = token.lower()
                vocab_count.setdefault(token, 0)
                vocab_count[token]+=1

        self.all_labels = list(set([mention['label'] for mention in mentions_all]))
        self.all_labels.sort()

        _START_VOCAB = ['_PAD','_UNK','_EOS']
        self.vocab   = _START_VOCAB + sorted(vocab_count, key=vocab_count.get, reverse=True)
        self.embed   = np.asarray([self.word2vec[w] if w in self.word2vec else np.zeros(self.word_embed_dim) for w in self.vocab])

        print('There are %d tokens in all.' % len(self.vocab))
        print('Max length of tokenized sentences is %d' % (self.max_length))
        assert(self.padded_length > self.max_length)
        return self

