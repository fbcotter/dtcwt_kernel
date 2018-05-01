import tensorflow as tf
import numpy as np
import py3nvml

import pytest
import os

LIB_BASE = os.path.join(os.path.dirname(__file__), '..', 'lib')

def test_kernel():
    py3nvml.grab_gpus(0)
    kernel_module = tf.load_op_library(os.path.join(LIB_BASE, 'kernel_example.so'))
    with tf.Session() as sess:
        result = kernel_module.example([5, 4, 3, 2, 1])
        np.testing.assert_array_equal(result.eval(), np.array([10, 8, 6, 4, 2]))


if __name__ == '__main__':
    pytest.main([__file__])
