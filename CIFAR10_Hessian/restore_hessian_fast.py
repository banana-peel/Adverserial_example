
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

def make_hparam_string(sel):
    return str(sel)

options = {1:range(400), 2:range(400,800), 3:range(800,1200), 4:range(1200,1600), 5:range(1600,2000), 6:range(2000,2400), 7:range(2400,2800), 8:range(2800,3072)}


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

def cifar(ran):
    tf.reset_default_graph()
    # In[ ]:

    sel = 4011
    hess_op = []
    im_test = np.expand_dims(images_test[sel],axis=0)
    lb_test = np.expand_dims(labels_test[sel],axis=0)
    # start_time = time.time()
        
    #    data = tf.placeholder(tf.float32,[17,150*150])   #will probably throw an error here

    # writer = tf.summary.FileWriter("fin_run/cifar_graph_train2")
    # merged_summary = tf.summary.merge_all()

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
        for i in ran:
            ddx1 = tf.squeeze(tf.gradients(hess[i],x))
            ddx = tf.reshape(ddx1,[-1])
            hess_op.append(ddx)
            # tf.summary.histogram("hess",ddx)
            # merged_summary = tf.summary.merge_all()
            # if i%10 ==0:
            #     s = session.run(merged_summary,feed_dict={x: im_test, y:lb_test, drop:1})
            #     writer.add_summary(s,i)
     #       print("2")
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

        # writer = tf.summary.FileWriter("fin_run/cifar_graph_train2")
        # merged_summary = tf.summary.merge_all()
    #    x_batch, y_true_batch = random_batch()
        batch_acc = session.run(grad,feed_dict={x: im_test, y:lb_test, drop:1})
        # end_time = time.time()
        # time_dif = end_time - start_time
        # print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
     #   print(session.run(cost,feed_dict={x: im_test, y:lb_test, drop:1}))
     #   print(batch_acc)
        hessian = session.run(hess_op,feed_dict={x: im_test, y:lb_test, drop:1})
        print(np.array(hessian))

        np.save("sav4011_hess/hessian"+hparam,np.array(hessian))
        del hessian
        del hess_op

        # eig_val,eig_vec = np.linalg.eig(np.array(hessian))
        # np.save("eigen_val",eig_val)
        # np.save("eigen_vec",eig_vec)


for ind in [1,2,3,4,5,6,7,8]:
    start_time = time.time()
    hparam = make_hparam_string(ind)
    ran = options[ind]
    cifar(ran)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
