
# coding: utf-8

# In[1]:

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os


img_size = 32
num_channels = 3
num_classes = 10


# In[18]:

images_train = np.load("images_train.npy")
labels_train = np.load("labels_train.npy")
images_test = np.load("images_test.npy")
labels_test = np.load("labels_test.npy")
# In[ ]:

train_batch_size = 68


def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

# def cifar(ran):
tf.reset_default_graph()
# In[ ]:

sel = 9885
hess_op = []
im_test = np.expand_dims(images_test[sel],axis=0)
lb_test = np.expand_dims(labels_test[sel],axis=0)


with tf.Session() as session:
    saver = tf.train.import_meta_graph('t_data/models/checkpoints2/checkpoints2-52000.meta')
    saver.restore(session,'t_data/models/checkpoints2/checkpoints2-52000')
    # saver = tf.train.import_meta_graph('checkpoints/checkpoints2-3.meta')
    # saver.restore(session,'checkpoints/checkpoints2-3')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y_true:0")
    drop = graph.get_tensor_by_name("dropout/drop:0")
    acc = graph.get_tensor_by_name("accuracy/accuracy:0")
    cost = graph.get_tensor_by_name("loss/loss:0")
    grad = tf.squeeze(tf.gradients(cost,x))
    hess = tf.reshape(grad,[-1])

    gradient = session.run(hess,feed_dict={x: im_test, y:lb_test, drop:1})
    print(np.array(gradient))

    # np.save("sav9301_hess/gradient",np.array(gradient))


    k = np.linspace(-0.3,0.3,num=1000)
#print(k)
    print(np.std(gradient))
    plt.hist(gradient, bins=k)
    plt.title("Histogram of gradient for Image: 9301")
    plt.show()