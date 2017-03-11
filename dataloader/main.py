# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from SemevalDataLoader import SemevalDataLoader
from SemEvalVectorizer import SemEvalVectorizer

import numpy as np
if __name__ == '__main__':
    import yaml
    conf = yaml.load(open('./semeval.yml'))

    MAKE_DATA = False
    if MAKE_DATA:
        dataloader = SemevalDataLoader(conf['Dataloader'])
        mentions_train, mentions_test = dataloader.get_processed_data()

        vectorizer = SemEvalVectorizer(conf['Vectorizer'])
        vectorizer.fit(mentions_train + mentions_test)
        train_x, train_y, train_pf_e1, train_pf_e2 = vectorizer.transform(mentions_train)
        test_x,  test_y,   test_pf_e1,  test_pf_e2 = vectorizer.transform(mentions_test)

        import os
        # save embed
        embed_name = vectorizer.word2vec_path.split(os.sep)[-1]
        np.save('../../corpus/SemEval2010/%s.npy' % embed_name, vectorizer.embed)

        # save training data
        np.save('../../corpus/SemEval2010/train_x.npy', train_x)
        np.save('../../corpus/SemEval2010/train_y.npy', train_y)
        np.save('../../corpus/SemEval2010/train_pf_e1.npy', train_pf_e1)
        np.save('../../corpus/SemEval2010/train_pf_e2.npy', train_pf_e2)

        # save test data
        np.save('../../corpus/SemEval2010/test_x.npy', test_x)
        np.save('../../corpus/SemEval2010/test_y.npy', test_y)
        np.save('../../corpus/SemEval2010/test_pf_e1.npy', test_pf_e1)
        np.save('../../corpus/SemEval2010/test_pf_e2.npy', test_pf_e2)

    train_x     = np.load('../../corpus/SemEval2010/train_x.npy')
    train_y     = np.load('../../corpus/SemEval2010/train_y.npy')
    train_pf_e1 = np.load('../../corpus/SemEval2010/train_pf_e1.npy')
    train_pf_e2 = np.load('../../corpus/SemEval2010/train_pf_e2.npy')

    test_x     = np.load('../../corpus/SemEval2010/test_x.npy')
    test_y     = np.load('../../corpus/SemEval2010/test_y.npy')
    test_pf_e1 = np.load('../../corpus/SemEval2010/test_pf_e1.npy')
    test_pf_e2 = np.load('../../corpus/SemEval2010/test_pf_e2.npy')
