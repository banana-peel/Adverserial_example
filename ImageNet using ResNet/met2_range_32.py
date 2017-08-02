from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import tensorflow as tf

import numpy as np

from datasets import dataset_factory

from deployment import model_deploy

from nets import nets_factory

from preprocessing import preprocessing_factory

from tensorflow.core.framework import graph_pb2

from tensorflow.core.protobuf import saver_pb2

from tensorflow.python.client import session

from tensorflow.python.framework import graph_io

from tensorflow.python.framework import importer

from tensorflow.python.framework import ops

from tensorflow.python.framework import test_util

from tensorflow.python.ops import math_ops

from tensorflow.python.ops import variables
from tensorflow.python.platform import test


from tensorflow.python.tools import freeze_graph

from tensorflow.python.training import saver as saver_lib

import time
from datetime import timedelta

slim = tf.contrib.slim


image = 4
images_test = np.load("test_slm/rnet_ip_img_rep.npy")
labels_test = np.load("test_slm/rnet_ip_lab.npy")


hess_op = []
# im_test = np.expand_dims(images_test[:32],axis=0)
# lb_test = np.expand_dims(labels_test[:32],axis=0)

im_test = images_test[:32]
lb_test = labels_test[:32]


def make_hparam_string(sel):
	print("current run :",sel)
	return str(sel)

# writer = tf.summary.FileWriter("resnet_gph")





def cifar_run(sel):
	tf.reset_default_graph()
	output_graph_def = graph_pb2.GraphDef()
	with open('test_slm/opti_rnet_gph.pb', "rb") as f:
	        output_graph_def.ParseFromString(f.read())
	        _ = importer.import_graph_def(output_graph_def, name="")
	p = [n.name for n in output_graph_def.node]
	q = p[2:]
	#print(*q, sep='\n')

	trunc_graph = tf.graph_util.extract_sub_graph(output_graph_def,q)



	writer = tf.summary.FileWriter("new_gph")
	with tf.Graph().as_default() as g_combined:
		inpt = tf.placeholder(tf.float32, (32, 224,224,3), name='inpt_img')
		lab_inpt = tf.placeholder(tf.float32,(32,5), name='inpt_labels')
		epsilon = 0.01*tf.ones([224,224,3])
		hard_max = tf.ones([224,224,3])
		hard_min = tf.zeros([224,224,3])
		alpha = 0.003

		loss,logit, = tf.import_graph_def(trunc_graph, input_map={"fifo_queue_Dequeue:0": inpt, 'fifo_queue_Dequeue:1':lab_inpt},\
			return_elements=["softmax_cross_entropy_loss/value:0","resnet_v2_50/SpatialSqueeze:0"])
		image_max = epsilon + inpt[sel]  #This will throw an error need to change the dimension of epsilon to include training set as well
		image_min = inpt[sel] - epsilon

		# logit, = tf.import_graph_def(trunc_graph, input_map={"fifo_queue_Dequeue:0": inpt, 'fifo_queue_Dequeue:1':lab_inpt},\
		# 	return_elements=["resnet_v2_50/SpatialSqueeze:0"])
		with tf.Session() as sess:
			# start_time = time.time()
			p_out = sess.run(loss,feed_dict={inpt: im_test, lab_inpt:lb_test})
			for ind in range(10):
				grad = tf.squeeze(tf.gradients(loss,inpt))[sel]
				sign = tf.sign(grad)
				perturb = alpha*sign
				per_fin = perturb
				test_image = per_fin + inpt[sel]
				clip1 = tf.maximum(hard_min,image_min)
				clip2 = tf.maximum(clip1,test_image)
				clip3 = tf.minimum(clip2,hard_max)
				clip = tf.minimum(clip3,image_max)
				op_image = clip
				sub_img = tf.expand_dims(clip,0)
				# feedbk = tf.concat([sub_img,inpt[1:]],0)
				# hess = tf.reshape(grad,[-1])
				# ddx1 = tf.squeeze(tf.gradients(hess[0],inpt))
				tm_ip = sess.run(sub_img,feed_dict={inpt: im_test, lab_inpt:lb_test})
				im_test[sel] = tm_ip[0]
			c_out = sess.run(loss,feed_dict={inpt: im_test, lab_inpt:lb_test})
			delta = c_out - p_out
			end_time = time.time()
			time_dif = end_time - start_time
			# print(delta)
			# print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
	return(delta)

delt = np.zeros(32)
# input_index = [9301, 2, 8235, 9885, 7021, 4011]
for sel in range(32):
	start_time = time.time()
	delt[sel] =cifar_run(sel)
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
print(delt)
np.save("test_slm/del_mat_rnet",delt)

