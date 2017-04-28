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
#score = loaded_model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#data = tf.variable(tf.zeros([x_test])) #might wanna change this
#data_out = loaded_model(data)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(data_out,y_test))
#ip_grad = tf.gradients(cost,data)[0]
#sig_grad = tf.sign(ip_grad)
#variable init
#init_op = tf.global_variables_initializer()

# Later, when launching the model
#sess = tf.Session()
  # Run the init operation.
#sess.run(init_op)
#sess.run(data_out,cost,ip_grad)
#sess.close()

#tf.Print(sig_grad)

data = tf.placeholder(tf.float32,[10000,32,32,3])
xp_test = tf.expand_dims(data[0],0)
print(xp_test.shape,'testing the shape')

#epsilon = tf.constant(.1, tf.float32)
#data = tf.placeholder(tf.float32,[1,32,32,3])
data_label = tf.placeholder(tf.float32)
data_out = loaded_model(xp_test)
#data_out = loaded_model.predict(xp_test)
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
#score = loaded_model.evaluate(test_image_p, y_test[1], verbose=0)
adv_op = loaded_model(test_image_p)
orig_op = loaded_model(xp_test)
#print(sess.run(adv_op, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0}))
#print(sess.run(orig_op, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0}))
print(y_test[0])


per_out = tf.cast(sign[0,0,:],tf.uint8)
orig_out = tf.cast(x_test[0]*255,tf.uint8)
adv_out = tf.cast(test_image*255,tf.uint8)

per_enc = tf.image.encode_jpeg(per_out)
orig_enc= tf.image.encode_jpeg(orig_out)
adv_enc	= tf.image.encode_jpeg(adv_out)

fwrite1 = tf.write_file('perturbation.jpg', per_enc)
fwrite2 = tf.write_file('original.jpg', orig_enc)
fwrite3 = tf.write_file('adverserial.jpg', adv_enc)

sess.run(fwrite1, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0})
sess.run(fwrite2, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0})
sess.run(fwrite3, feed_dict={data:x_test,data_label:y_test[0], K.learning_phase(): 0})


#print("%.2f%%" % (score*100))
