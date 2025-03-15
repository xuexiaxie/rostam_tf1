import networkx as nx
import json

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer
import graph_profile_pb2

import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim

# 创建计算图
with tf.device('/GPU:0'):
    a = tf.constant([1.0], shape=[1, 1], name='input_a')
    b = tf.constant([2.0], shape=[1, 1], name='input_b')
    c = tf.matmul(a, b, name='output_c')

# 运行 Profiler
with tf.Session() as sess:
    run_metadata = tf.RunMetadata()
    options = tf.profiler.ProfileOptionBuilder.float_operation()
    profiler_result = tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='scope',
        options=options
    )
    result = sess.run(c)
    print("计算结果:", result)
    print("Profiler 统计:")
    print(profiler_result)