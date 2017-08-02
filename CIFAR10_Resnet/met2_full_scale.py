import tensorflow as tf
import numpy as np
from resnet import *
import time
from datetime import timedelta

img_size = 32
num_channels = 3
num_classes = 10

hess_op=[]

# In[18]:


images_test = np.load("images_test.npy")
labels_test = np.load("labels_test.npy")


img_size = 32
num_channels = 3
num_classes = 10

def make_hparam_string(sel):
	print("current run :",sel)
	return str(sel)



def cifar_run(sel):
	tf.reset_default_graph()
	epsilon = 0.01*tf.ones([32,32,3])
	image = 4
	hard_max = tf.expand_dims(tf.ones([32,32,3]),0)
	hard_min = tf.expand_dims(tf.zeros([32,32,3]),0)
	alpha = 0.003

	hess_op = []
	im_test = np.expand_dims(images_test[sel],axis=0)
	lb_test = np.expand_dims(labels_test[sel],axis=0)

	# im_test = images_test[2:7]
	# lb_test = labels_test[2:7]

	input_img = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3])
	true_lab = tf.placeholder(dtype=tf.float32, shape=[None,10])


	logits = inference(input_img,18,reuse=False)
	predictions = tf.nn.softmax(logits)
	correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(true_lab, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

	image_max = tf.expand_dims(epsilon,0) + input_img  #This will throw an error need to change the dimension of epsilon to include training set as well
	image_min = input_img - tf.expand_dims(epsilon,0)


	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_lab, logits=logits),name="loss")



	# grad = tf.squeeze(tf.gradients(cross_entropy,input_img))
	saver = tf.train.Saver(tf.global_variables())
	# writer = tf.summary.FileWriter("net_run/cifar"+hparam)

	with tf.Session() as session:
		saver.restore(session,"./model_110.ckpt-79999")
		p_out=session.run(cross_entropy,feed_dict={input_img: im_test, true_lab:lb_test})

		for ind in range(20):
			grad = tf.gradients(cross_entropy,input_img)
			sign = tf.sign(grad)
			perturb = alpha*sign
			per_fin = perturb[0]
			test_image = per_fin + input_img
			clip1 = tf.maximum(hard_min,image_min)
			clip2 = tf.maximum(clip1,test_image)
			clip3 = tf.minimum(clip2,hard_max)
			clip = tf.minimum(clip3,image_max)
			op_image = clip[0]
			tm_ip = session.run(clip,feed_dict={input_img: im_test, true_lab:lb_test})
			# print(tm_ip)
			im_test = tm_ip
		c_out = session.run(cross_entropy,feed_dict={input_img: im_test, true_lab:lb_test})
		delta = c_out - p_out
		# print(accuracy,session.run(accuracy,feed_dict={input_img: im_test, true_lab:lb_test}))
		# print(lb_test)
		# print(session.run(predictions,feed_dict={input_img: im_test, true_lab:lb_test}))
	return(delta)

delt = np.zeros(6)
input_index = [9301, 2, 8235, 9885, 7021, 4011]
for sel in range(6):
	start_time = time.time()
	delt[sel] =cifar_run(input_index[sel])
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
print(delt)
np.save("del_mat_rnet",delt)