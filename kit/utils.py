# -*- coding: utf-8 -*-
# File: utils.py
# Author: Tianjian Jiang <jiangtj13@gmail.com>

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import kit.flags

FLAGS = kit.flags.FLAGS

def report_func(epoch, 
                batch_idx,
                num_batches,
                start_time, 
                loc_loss,
                clf_loss):
    # TODO: try to make it more abstract
    if ((batch_idx % FLAGS.interval) == (FLAGS.interval - 1)):
        print("Iter: {:3d}, Batch: {:5d}/{:5d}, time: {:5.4f} loc_loss: {:.4f}, clf_loss: {:.4f} \n".format(epoch, batch_idx + 1, num_batches, time.time() - start_time, loc_loss, clf_loss))
        sys.stdout.flush()
    return



