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
# # -----------------------
# # FAST GRADIENT
# data = tf.placeholder(tf.float32,[10000,32,32,3])

# #data = tf.placeholder(tf.float32,[10000,32,32,3])
# xp_test = tf.slice(data,[99,0,0,0],[100,32,32,3])
# print(xp_test.shape,'testing the shape')

# #epsilon = tf.constant(.1, tf.float32)
# #data = tf.placeholder(tf.float32,[1,32,32,3])
# data_label = tf.placeholder(tf.float32,[10000,10])
# data_label_slc = tf.slice(data_label,[99,0],[100,10])
# data_out = loaded_model(xp_test)
# #data_out = loaded_model.predict(xp_test)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =data_label_slc, logits = data_out))
# grad = tf.gradients(cost,xp_test)
# sign = tf.sign(grad)
# perturb = 0.01 * sign
# print(sess.run(perturb,feed_dict={data:x_test,data_label:y_test, K.learning_phase(): 0}))
# print(sign.shape,'shape of image')
# per_fin = tf.reduce_mean(perturb,0)
# print(per_fin.shape, 'shape of perturbation')



# #checking the accuracy

# test_image = per_fin + xp_test
# #test_image_p = tf.expand_dims(test_image,0)
# #score = loaded_model.evaluate(test_image_p, y_test[1], verbose=0)
# adv_op = loaded_model(test_image)
# orig_op = loaded_model(xp_test)
# adv_lab = tf.argmax(tf.squeeze(adv_op),1)   #check this out if it has to be [1] or [0]
# orig_lab = tf.argmax(tf.squeeze(orig_op),1)
# acc_lab = tf.argmax(data_label_slc,1)
# accuracy_orig = tf.reduce_sum(tf.cast(tf.equal(orig_lab,acc_lab),tf.float32))
# accuracy_adv =	tf.reduce_sum(tf.cast(tf.equal(adv_lab,acc_lab),tf.float32)) 
# print(sess.run(accuracy_orig, feed_dict={data:x_test,data_label:y_test, K.learning_phase(): 0}))
# print(sess.run(accuracy_adv, feed_dict={data:x_test,data_label:y_test, K.learning_phase(): 0}))



# print("%.2f%%" % (score*100))
# print("%.2f%%" % (score*100))



epsilon = 7
b_init = numpy.zeros(input_shape)

b = b_init
pred_x = loaded_model.predict(numpy.expand_dims(x_test[0], axis=0))
pred_perturb = loaded_model.predict(numpy.expand_dims((x_test[0] + b), axis=0))
J_init = numpy.sum(pred_x*numpy.log2(pred_perturb))
print('J_init: ', J_init)

test_range = 10
# mini_batch = 30

J = numpy.zeros(test_range)
b = numpy.random.random(input_shape)
b = epsilon*b/numpy.linalg.norm(b)

# Let's try scipy optimize minimize

def cross_ent(b):
    b = numpy.reshape(b,input_shape)
    J = 0
    for iter in range(0,test_range):
        pred_x = loaded_model.predict(numpy.expand_dims(x_test[offset + iter], axis=0))
        pred_perturb = loaded_model.predict(numpy.expand_dims((x_test[offset + iter] + b), axis=0))

        # Noise term to avoid zero error in the log function
        avoid_zeros = 0.0000000001*numpy.ones(pred_perturb.shape)

        y_log_p = pred_x*numpy.log2(pred_perturb + avoid_zeros)
        one_minus = (1-pred_x)*numpy.log2((1-pred_perturb) + avoid_zeros)

        J = J + numpy.sum(y_log_p + one_minus)
    return J/test_range


def constraint(b):
    return numpy.linalg.norm(b) - epsilon

cons = ({'type': 'ineq', 'fun': constraint })

x_mini_test = x_test[0:test_range]
x_mini_perturbed = x_test[0:test_range]
y_mini_test = y_test[0:test_range]

print('Performance for unperturbed data:')
new_score = loaded_model.evaluate(x_mini_test, y_mini_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], new_score[1]*100))





epochs = 10
for i in range(0, epochs):
    print('Starting ',i)
    offset = i * 10;
    # sol = spo.minimize(cross_ent, b_init, constraints=cons)
    sol = spo.minimize(cross_ent, b_init, method='BFGS')
    b_init = epsilon*sol.x/numpy.linalg.norm(sol.x)


    print('Iteration ',i, ' completed.')
    for k in range(0,test_range):
        x_mini_perturbed[k] = x_mini_perturbed[k] + numpy.reshape(b, input_shape)

    # Testing the perturbation in the network
    print('Performance for perturbed data:')
    new_score = loaded_model.evaluate(x_mini_perturbed, y_mini_test, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], new_score[1]*100))

    print('Norm of perturbation', numpy.linalg.norm(b))
    print('----')
