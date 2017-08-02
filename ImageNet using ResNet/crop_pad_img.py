#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 00:38:38 2017

@author: saurabh
"""

import tensorflow as tf
import numpy as np
import skimage.io as io

new_store = []
IMAGE_DIM = 224

inp_label = np.load("test_slm/val_lab.npy")
inp_img = np.load("test_slm/val_img.npy")

g = tf.placeholder(tf.uint8,(None,None,3))
h = tf.placeholder(tf.int32,(350))
resized_image = tf.image.resize_image_with_crop_or_pad(image=g,\
                                                       target_height=IMAGE_DIM,\
                                                       target_width=IMAGE_DIM)



with tf.Session() as sess:
    lab = tf.one_hot(h,5,axis=-1)
    new_label = sess.run(lab,feed_dict={h:np.array(inp_label)})
    for i in range(350):
        new_img = (np.array(sess.run(resized_image,feed_dict={g:inp_img[i]})))/255
        new_store.append(new_img)
    np.save("test_slm/rnet_ip_img",new_store)
    np.save("test_slm/rnet_ip_lab",new_label)