from __future__ import print_function
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.datasets import cifar10
import numpy as np
import os



sess = tf.Session()
K.set_session(sess)
hess_op = []



X_train = np.load('xtrain.npy')
Y_train = np.load('ytrain.npy')
X_test = np.load('xtest.npy')
Y_test = np.load('ytest.npy')



print("labels",Y_test)
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# load json and create model
json_file = open('yf_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("yf_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


data = tf.placeholder(tf.float32,[17,150*150])   #will probably throw an error here
xp_test1 = data[0]
xp_test = tf.expand_dims(xp_test1,0)
print(xp_test.shape,'testing the shape')

data_label1 = tf.placeholder(tf.float32,[17,15])
data_label = tf.reshape(data_label1[0],[1,15])
data_out = loaded_model(xp_test)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =data_label, logits = data_out))
grad = tf.gradients(cost,xp_test)
sign = tf.sign(grad)
perturb = 0.1 * sign
print(sess.run(perturb,feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0}))
print(sign.shape,'shape of image')
per_fin = perturb[0,0,:]
print(per_fin.shape, 'shape of perturbation')



#checking the accuracy

test_image = per_fin + xp_test
test_image_p = tf.expand_dims(test_image,0)
adv_op = loaded_model(test_image_p)
orig_op = loaded_model(xp_test)
adv_op1 = tf.argmax(tf.squeeze(adv_op),0)
orig_op1 = tf.argmax(tf.squeeze(orig_op),0)
print(sess.run(adv_op, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0}))
print(sess.run(orig_op, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0}))
print(sess.run(data_label, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0}))
print(sess.run(adv_op1, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0}))
print(sess.run(orig_op1, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0}))

hess = tf.squeeze(tf.gradients(cost,xp_test1))
print(hess.shape,"Shape of 1st derivative")
for i in range(22500):
	ddx = tf.gradients(hess[i],xp_test)[0]
	hess_op.append(ddx)


eig_ip = sess.run(hess_op, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0})
eig_val,eig_vec = numpy.linalg.eig(eig_ip)
np.save('eigval',eig_val)
print(eig_val,'Eigen Values')

