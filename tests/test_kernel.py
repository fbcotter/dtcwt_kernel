import tensorflow as tf
import numpy as np
import py3nvml

def test_zeroout():
    py3nvml.grab_gpus(0)
    zero_out_module = tf.load_op_library('../bin/zero_out.so')
    with tf.Session() as sess:
        result = zero_out_module.zero_out([5, 4, 3, 2, 1])
        np.testing.assert_array_equal(result.eval(), np.array([5,0,0,0,0]))


