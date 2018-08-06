from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import numpy as np
import tensorflow as tf


class ItemPool(object):

    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):

        recyclePreviousSamplesWithProbability = 0.5

        """ in_items is a list of item"""
        if self.max_num == 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else: #Every second time, we store a copy of a random existing item, and otherwise, we add the new item. This was already in the original PyToch impl. and it follows Shrivastava 2017 (SimGAN) idea.
                if np.random.rand() > 1.0 - recyclePreviousSamplesWithProbability:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)

        return return_items


def mkdir(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)


def load_checkpoint(checkpoint_dir, sess, saver):
    print(" [*] Loading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        saver.restore(sess, ckpt_path)
        print(" [*] Loading successful!")
        return ckpt_path
    else:
        print(" [*] No suitable checkpoint!")
        return None
