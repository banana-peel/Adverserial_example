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
import numpy
import os

sess = tf.Session()
K.set_session(sess)
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape, 'test samples')

hess_op=[]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# load json and create model
json_file = open('cifar_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cifar_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


data = tf.placeholder(tf.float32,[10000,32,32,3])
hess_ip = tf.reshape(data[0],[-1])
print(hess_ip.shape, 'hessian input shape')
xp_test1 = tf.reshape(hess_ip,[32,32,3])
xp_test = tf.expand_dims(xp_test1,0)
print(xp_test.shape,'testing the shape')



data_label = tf.placeholder(tf.float32)
data_out = loaded_model(xp_test)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =data_label, logits = data_out))
grad = tf.gradients(cost,xp_test)
sign = tf.sign(grad)
perturb = 0.1 * sign
print(sess.run(perturb,feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0}))
print(sign.shape,'shape of image')
per_fin = perturb[0,0,:]
print(per_fin.shape, 'shape of perturbation')



#checking the accuracy

test_image = per_fin + x_test[0]
test_image_p = tf.expand_dims(test_image,0)
adv_op = loaded_model(test_image_p)
orig_op = loaded_model(xp_test)

#hess = tf.hessians(cost,hess_ip)
#print(sess.run(hess, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0}))

hess = tf.gradients(cost,hess_ip)[0]
for i in range(3072):
	ddx = tf.gradients(hess[i],hess_ip)[0]
	hess_op.append(ddx)


eig_ip = sess.run(hess_op, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0})
eig_val,eig_vec = numpy.linalg.eig(eig_ip)
print(eig_val,'Eigen Values')

