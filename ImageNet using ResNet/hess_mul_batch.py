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

import sys


slim = tf.contrib.slim


image = 4
images_test = np.load("test_slm/rnet_ip_img_rep.npy")
labels_test = np.load("test_slm/rnet_ip_lab.npy")

fin_hess=[]


# im_test = np.expand_dims(images_test[:32],axis=0)
# lb_test = np.expand_dims(labels_test[:32],axis=0)

im_test = images_test[:32]
lb_test = labels_test[:32]


def make_hparam_string(sel):
	print("current run :",sel)
	return str(sel)

# writer = tf.summary.FileWriter("resnet_gph")


index = (np.reshape(np.linspace((sys.argv[1]*20+1),(sys.argv[1]*20+20),20,dtype=np.int32),(-1,10)))-1
index = list(index)





def cifar_run(ran):
	start_time = time.time()
	tf.reset_default_graph()
	hess_op = []
	output_graph_def = graph_pb2.GraphDef()
	with open('test_slm/opti_rnet_gph_ndv.pb', "rb") as f:
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


		loss,logit, = tf.import_graph_def(trunc_graph, input_map={"fifo_queue_Dequeue:0": inpt, 'fifo_queue_Dequeue:1':lab_inpt},\
			return_elements=["softmax_cross_entropy_loss/value:0","resnet_v2_50/SpatialSqueeze:0"])


		# logit, = tf.import_graph_def(trunc_graph, input_map={"fifo_queue_Dequeue:0": inpt, 'fifo_queue_Dequeue:1':lab_inpt},\
		# 	return_elements=["resnet_v2_50/SpatialSqueeze:0"])
		with tf.Session() as sess:
			grad = tf.squeeze(tf.gradients(loss,inpt))[image]
			grad = tf.reshape(grad,[-1])
			for i in ran:
				ddx1 = tf.squeeze(tf.gradients(grad[i],inpt))
				ddx1 = tf.reshape(ddx1[image],[-1])
				hess_op.append(ddx1)
			hessian=np.array(sess.run(hess_op,feed_dict={inpt: im_test, lab_inpt:lb_test}))
			print(hessian.shape)
			del hess_op
			del ddx1
	return(hessian)

# delt = np.zeros(32)
# # input_index = [9301, 2, 8235, 9885, 7021, 4011]
# for sel in range(32):
# 	start_time = time.time()
# 	delt[sel] =cifar_run(sel)
	# end_time = time.time()
	# time_dif = end_time - start_time
	# print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
# print(delt)
# np.save("test_slm/del_mat_rnet",delt)





for ind in index:	
	start_time = time.time()
	hparam = make_hparam_string(ind[0])
	otput =  cifar_run(ind)
	fin_hess.append(otput)
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
np.save("uest_slm/hess"+hparam,np.array(fin_hess))

