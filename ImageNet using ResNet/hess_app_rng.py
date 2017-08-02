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



images_test = np.load("test_slm/rnet_ip_img_rep.npy")
labels_test = np.load("test_slm/rnet_ip_lab.npy")

eig_hess=[]

# im_test = np.expand_dims(images_test[:32],axis=0)
# lb_test = np.expand_dims(labels_test[:32],axis=0)

im_test = images_test[:32]
lb_test = labels_test[:32]


def make_hparam_string(sel):
	print("current run :",sel)
	return str(sel)

# writer = tf.summary.FileWriter("resnet_gph")





def cifar_run(image):
	# start_time = time.time()
	tf.reset_default_graph()
	hess_op = []
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


		loss,logit, = tf.import_graph_def(trunc_graph, input_map={"fifo_queue_Dequeue:0": inpt, 'fifo_queue_Dequeue:1':lab_inpt},\
			return_elements=["softmax_cross_entropy_loss/value:0","resnet_v2_50/SpatialSqueeze:0"])
		op_prob = tf.nn.softmax(logit)
		cross_entpy = -(tf.add(tf.multiply(tf.log(op_prob),lb_test),tf.multiply(tf.log(1-op_prob),(1-lb_test))))


		with tf.Session() as sess:
			perturb = 0
        #cross_entpy = -(tf.multiply(tf.log(i), true_lab))
	        for iter in range(20):
	            #randomly perturb the data


	                #print(session.run(op_prob))
	            grad = tf.squeeze(tf.gradients((cross_entpy),inpt))
	            hess = tf.reshape(grad[image],[-1])
	            inp_moment=sess.run(hess,feed_dict={inpt: im_test+perturb, lab_inpt:lb_test})
	            hess_op.append(inp_moment)
	            sign = np.random.choice([-1,1])
	            perturb = 0.003*sign
	        derv = np.array(hess_op)
	        np.array(hess_op)
	return(derv)


for ind in range(32):
	start_time = time.time()
	inpt = cifar_run(ind)
	print(inpt.shape)
	inpt_sq = np.square(inpt)
	scnd_moment = np.average(inpt_sq,axis=0)
	fst_moment = np.square(np.average(inpt,axis=0))
	print(fst_moment.shape)
	eig_val = scnd_moment - fst_moment
	print(np.std(eig_val))
	eig_hess.append(eig_val)
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
np.save("test_slm/eig_val_list",eig_hess)

# plt.hist(eig_val, bins=k)
# print(np.std(eig_val))


