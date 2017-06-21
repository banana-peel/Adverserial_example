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

# In[3]:

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[4]:

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name="W")

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name="b")


# In[5]:

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


# In[6]:

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[7]:

with tf.name_scope("conv1"):
    W_conv1 = weight_variable([5, 5, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)


# In[8]:

with tf.name_scope("conv2"):
    W_conv2 = weight_variable([5, 5, 16, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    siz = tf.shape(h_conv2)


# In[9]:

with tf.name_scope("nn1"):
    W_fc1 = weight_variable([32 * 32 * 16, 1024])
    b_fc1 = bias_variable([1024])
    h_conv2_flat = tf.reshape(h_conv2, [-1, 32*32*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)


# In[10]:

with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32,name="drop")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[11]:

with tf.name_scope("nn2"):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2,name="pl_out")


# In[12]:

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv),name="loss")


# In[13]:

with tf.name_scope("optimize"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train")


# In[14]:

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")


# In[15]:

num_iterations = 4

tf.summary.scalar("cross_entropy",cross_entropy)
tf.summary.scalar("acc",accuracy)
merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("cifar_graph_train2")
with tf.Session() as session:
    writer.add_graph(session.graph)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
#        feed_dict_train = {x: x_batch,
#                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
#        i_global, _ = session.run([global_step, optimizer],
#                                  feed_dict=feed_dict_train)
#
        train_step.run(feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.5})
        
        
        if i % 50 ==0:
            s= session.run(merged_summary,feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.5})
            writer.add_summary(s,i)

            
        # Print status to screen every 100 iterations (and last).
        if (i % 4000 == 0) or (i == num_iterations - 1):
            saver.save(session,save_path="checkpoints/checkpoints2",global_step=i)
            # Calculate the accuracy on the training-batch.
#             batch_acc = session.run(accuracy,feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.5})

#             # Print status.
# #            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
#             print(batch_acc)
    end_time = time.time()
    saver.save(session,save_path="checkpoints/checkpoints2",global_step=1111)

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
