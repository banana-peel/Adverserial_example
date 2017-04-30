from __future__ import print_function
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.image as mpimg
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.datasets import cifar10
import os

import cv2
import numpy as np
from skimage import io
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


sess = tf.Session()
K.set_session(sess)
hess_op = []


DatasetPath = []
for i in os.listdir("yalefaces"):
    DatasetPath.append(os.path.join("yalefaces", i))

imageData = []
imageLabels = []

for i in DatasetPath:
    imgRead = io.imread(i,as_grey=True)
    imageData.append(imgRead)
    
    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
    imageLabels.append(labelRead)

faceDetectClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imageDataFin = []
for i in imageData:
    facePoints = faceDetectClassifier.detectMultiScale(i)
    x,y = facePoints[0][:2]
    cropped = i[y: y + 150, x: x + 150]
    imageDataFin.append(cropped)

c = np.array(imageDataFin)
c.shape


X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.9, random_state = 20)

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train.shape

X_test.shape

nb_classes = 15
y_train = np.array(y_train) 
y_test = np.array(y_test)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.reshape(149, 150*150)
X_test = X_test.reshape(17, 150*150)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

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

hess = tf.squeeze(tf.hessians(cost,xp_test1))
print(hess.shape,"Shape of 1st derivative")
#for i in range(22500):
#	ddx = tf.gradients(hess[i],xp_test)[0]
#	hess_op.append(ddx)


eig_ip = sess.run(hess, feed_dict={data:X_test,data_label1:Y_test, K.learning_phase(): 0})
eig_val,eig_vec = numpy.linalg.eig(eig_ip)
print(eig_val,'Eigen Values')

