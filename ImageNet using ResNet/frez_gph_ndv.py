
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


slim = tf.contrib.slim
writer = tf.summary.FileWriter("resnet_gph")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('test_slm/flowers/resnet/model.ckpt-1001.meta')
    saver.restore(sess,'test_slm/flowers/resnet/model.ckpt-1001')
    writer.add_graph(sess.graph)
    graph_io.write_graph(sess.graph, "test_slm", "resnet_test_gph.pb")
    p = [n.name for n in tf.get_default_graph().as_graph_def().node]
    q = p[391:3722]
    #print(*q, sep='\n')
    input_graph_path = "test_slm/resnet_test_gph.pb"
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "softmax_cross_entropy_loss/value"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = "uest_slm/opti_rnet_gph_ndv.pb"
    clear_devices = True
    checkpoint_path = "test_slm/flowers/resnet/model.ckpt-1001"
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,\
    input_binary, checkpoint_path, output_node_names,\
    restore_op_name, filename_tensor_name,\
    output_graph_path, clear_devices, "")
