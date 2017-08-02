import tensorflow as tf
import numpy as np
import time
from datetime import timedelta

img_size = 32
num_channels = 3
num_classes = 10

# In[18]:

def make_hparam_string(sel):
    return str(sel)

index = (np.reshape(np.linspace(1,3070,3070,dtype=np.int32),(-1,10)))-1
data_index = list(index)
data_index.append(np.array([3070, 3071]))


hess_op = []

images_test = np.load("images_test.npy")
labels_test = np.load("labels_test.npy")

sel = 4011
im_test = np.expand_dims(images_test[sel],axis=0)
lb_test = np.expand_dims(labels_test[sel],axis=0)



def cifar(ran):
	tf.reset_default_graph()
	hess_op = []
	with tf.Session() as session:
		saver = tf.train.import_meta_graph('checkpoints2-1111.meta')
		saver.restore(session,'checkpoints2-1111')
		graph = tf.get_default_graph()
		# mid_time = time.time()
		# print("Time usage: " + str(timedelta(seconds=int(round(mid_time - start_time)))))
		input_img = graph.get_tensor_by_name("input_img:0")
		true_lab = graph.get_tensor_by_name("true_lab:0")
		cross_entropy = graph.get_tensor_by_name("loss:0")
		# accuracy = graph.get_tensor_by_name("accuracy:0")
		# print("accuracy =",session.run(accuracy,feed_dict={input_img:images_test[0:1000,:,:,:], true_lab:labels_test[0:1000,:]}))
		grad = tf.squeeze(tf.gradients(cross_entropy,input_img))
		hess = tf.reshape(grad,[-1])
		for i in ran:
		    ddx1 = tf.squeeze(tf.gradients(hess[i],input_img))
		    ddx = tf.reshape(ddx1,[-1])
		    hess_op.append(ddx)
		hessian=session.run(hess_op,feed_dict={input_img: im_test, true_lab:lb_test})
		np.save("sav4011_hess/hessian"+hparam,np.array(hessian))
		del hess_op
		del ddx
# start_time = time.time()

for ind in range(60):	
	start_time = time.time()
	hparam = make_hparam_string(ind)
	ran = data_index[ind]
	cifar(ran)
	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

