import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import scipy.stats as sc
import scipy.stats as bernoulli
import matplotlib.pyplot as plt

img_size = 32
num_channels = 3
num_classes = 10

eig_arr = []

# In[18]:

def make_hparam_string(sel):
	return str(sel)

index = (np.reshape(np.linspace(1,3070,3070,dtype=np.int32),(-1,10)))-1
data_index = list(index)
data_index.append(np.array([3070, 3071]))


hess_op = []

images_test = np.load("images_test.npy")
labels_test = np.load("labels_test.npy")


# sel = 8235
# im_test = np.expand_dims(images_test[sel],axis=0)
# lb_test = np.expand_dims(labels_test[sel],axis=0)
#im_test = images_test[sel]
#lb_test = labels_test[sel]


def cifar(sel):
    tf.reset_default_graph()
    im_test = np.expand_dims(images_test[sel],axis=0)
    lb_test = np.expand_dims(labels_test[sel],axis=0)
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
        cross_entpy = -(tf.add(tf.multiply(tf.log(op_prob),true_lab),tf.multiply(tf.log(1-op_prob),(1-true_lab))))
        perturb = 0
        #cross_entpy = -(tf.multiply(tf.log(i), true_lab))
        for iter in range(20):
            #randomly perturb the data


                #print(session.run(op_prob))
            grad = tf.squeeze(tf.gradients((cross_entpy),input_img))
            hess = tf.reshape(grad,[-1])
            inp_moment=session.run(hess,feed_dict={input_img: im_test+perturb, true_lab:lb_test})
            hess_op.append(inp_moment)
            sign = np.random.choice([-1,1])
            perturb = 0.0008*sign
        derv = np.array(hess_op)
        # np.array(hess_op)
    return(derv)
# start_time = time.time()

for i in [9301, 2, 8235, 4011, 7021, 9885]:
    start_time = time.time()
    inpt = cifar(i)
    print(inpt.shape)
    #inpt = np.reshape(inpt,[1,200*10])
    inpt_sq = np.square(inpt)
    scnd_moment = np.average(inpt_sq,axis=0)
    fst_moment = np.square(np.average(inpt,axis=0))
    eig_val = scnd_moment - fst_moment
    eig_arr.append(eig_val)
    # print(eig_val)
    # k = np.linspace(-1e-2,1e-2,num=1000)
    # plt.hist(eig_val, bins=k)
    print(np.std(eig_val))
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
np.save("eig_mat_full",eig_arr)
# plt.title("Histogram of eigen values for Image: 8235")
# plt.show()
