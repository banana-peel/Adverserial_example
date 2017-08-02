#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:48:44 2017

@author: saurabh
"""

import tensorflow as tf
from datasets import flowers
import numpy as np
import skimage.io as io

val_img = []
val_lab = []
cnt = 0

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', 'test_slm/flowers')


provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

with tf.Session() as sess:
    sess.run([
               tf.local_variables_initializer(),
               tf.global_variables_initializer(),
       ])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      while not coord.should_stop():
          raw_labels = sess.run(label)
          val_lab.append(raw_labels)
          raw_img = sess.run(image)
          val_img.append(raw_img)
          cnt += 1
          if cnt >=350:
              break

    finally:
        coord.request_stop()
        coord.join(threads)
        np.save("test_slm/val_img",val_img)
        np.save("test_slm/val_lab",val_lab)


          