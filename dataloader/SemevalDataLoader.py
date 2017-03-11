# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from nltk.tokenize import TreebankWordTokenizer

class SemevalDataLoader(object):
    ''' Load, tokenize, extract entities & its position for SemEval2010 Dataset'''
    def __init__(self, config = {}):
        self.tokenizer = TreebankWordTokenizer().tokenize
        self.data_dir  = os.path.expanduser('~/dataset/SemEval2010')
        self.load_config(config)

    def load_config(self, config):
        for k,v in config.items():
            setattr(self, k, v)

    def get_processed_data(self):
        mentions_train = self.get_mention(os.path.join(self.data_dir, 'train.txt'))
        mentions_test = self.get_mention(os.path.join(self.data_dir, 'test.txt'))
        return mentions_train, mentions_test

    def get_mention(self, filename):
        mentions = []
        with open(filename) as f:
            for i, line in enumerate(f):
                # sentences
                if i % 4 == 0:
                    number, line = line.split('\t')
                    line = line.strip().strip('"')
                    sentence = line
                    e1 = line[line.find('<e1>'):line.find('</e1>')+5];
                    e2 = line[line.find('<e2>'):line.find('</e2>')+5];

                    line = line.replace(e1, ' ENTITY1 ')
                    line = line.replace(e2, ' ENTITY2 ')
                    line = ' '.join(line.split())

                    line = self.tokenizer(line)

                    pos_e1 = line.index('ENTITY1')
                    pos_e2 = line.index('ENTITY2')

                    e1 = e1[4:-5]
                    e2 = e2[4:-5]

                    tokens = [w.replace('ENTITY1',e1) for w in line]
                    tokens = [w.replace('ENTITY2',e2) for w in tokens]
                # relation
                if i % 4 == 1:
                    label = line.strip()

                if i % 4 == 3:
                    mention = { 'id': int(number), 'sentence': sentence,
                                'tokens': tokens,  'label': label,
                                'pos_e1':pos_e1,   'pos_e2':pos_e2,
                                'e1': e1,          'e2': e2}
                    mentions.append(mention)
        return mentions


