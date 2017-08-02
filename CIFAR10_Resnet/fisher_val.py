import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import scipy.stats as sc 
import matplotlib.pyplot as plt

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

sel = 9301
im_test = np.expand_dims(images_test[sel],axis=0)
lb_test = np.expand_dims(labels_test[sel],axis=0)



def cifar():
	tf.reset_default_graph()
	hess_op = []
	with tf.Session() as session:
		saver = tf.train.import_meta_graph('checkpoints2-1111.meta')
		saver.restore(session,'checkpoints2-1111')
		graph = tf.get_default_graph()
		# mid_time = time.time()
		# print("Time usage: " + str(timedelta(seconds=int(round(mid_time - start_time)))))
		# p = [n.name for n in tf.get_default_graph().as_graph_def().node]
		# print(*p, sep='\n')
		raw_op = graph.get_tensor_by_name("fc/add:0")
		op_prob = graph.get_tensor_by_name("op_prob:0")

		input_img = graph.get_tensor_by_name("input_img:0")
		true_lab = graph.get_tensor_by_name("true_lab:0")
		cross_entropy = graph.get_tensor_by_name("loss:0")
		# cross_entpy = -(tf.add(tf.multiply(tf.log(op_prob),true_lab),tf.multiply(tf.log(1-op_prob),(1-true_lab))))
		cross_entpy = -(tf.multiply(tf.log(op_prob),true_lab))    
		# accuracy = graph.get_tensor_by_name("accuracy:0")
		# print("accuracy =",session.run(accuracy,feed_dict={input_img:images_test[0:1000,:,:,:], true_lab:labels_test[0:1000,:]}))
		for i in range(10):
			grad = tf.squeeze(tf.gradients((cross_entpy[0,i]),input_img))
			hess = tf.reshape(grad,[-1])
			inp_moment=session.run(hess,feed_dict={input_img: im_test, true_lab:lb_test})
			hess_op.append(inp_moment)
		print(hess_op)
		derv = np.array(hess_op)
	return(derv)
# start_time = time.time()


inpt = cifar()
print(inpt.shape)
inpt_sq = np.square(inpt)
scnd_moment = np.average(inpt_sq,axis=0)
fst_moment = np.square(np.average(inpt,axis=0))
print(fst_moment.shape)
eig_val = scnd_moment - fst_moment	
k = np.linspace(-1e-7,1e-2,num=3000)
plt.hist(eig_val, bins=k)
print(np.std(eig_val))
plt.title("Histogram of eigen values for Image: 9301")
plt.show()

