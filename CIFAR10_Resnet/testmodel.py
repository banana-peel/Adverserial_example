import tensorflow as tf
import numpy as np
from resnet import *

img_size = 32
num_channels = 3
num_classes = 10


# In[18]:

images_train = np.load("images_train.npy")
labels_train = np.load("labels_train.npy")
images_test = np.load("images_test.npy")
labels_test = np.load("labels_test.npy")

input_img = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3],name="input_img")
true_lab = tf.placeholder(dtype=tf.float32, shape=[None,10],name="true_lab")


logits = inference(input_img,18,reuse=False)
predictions = tf.nn.softmax(logits,name="op_prob")
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(true_lab, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_lab, logits=logits),name="loss")

# saver = tf.train.Saver(tf.global_variables())

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=50)
	saver.restore(session,"model_110.ckpt-79999")
	print("accuracy =",session.run(accuracy,feed_dict={input_img:images_test[0:1000,:,:,:], true_lab:labels_test[0:1000,:]}))
	saver.save(session,save_path="checkpoints2",global_step=1111)
	# print(session.run(logits,feed_dict={input_img:images_test[0:2,:,:,:], true_lab:labels_test[0:2,:]}))
	# print(session.run(true_lab,feed_dict={input_img:images_test[0:2,:,:,:], true_lab:labels_test[0:2,:]}))