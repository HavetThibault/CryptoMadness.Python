import tensorflow as tf


class SigmoidActivation:
    def __init__(self, min_val, max_val):
        self._min_val = min_val
        self._max_val = max_val

    def activation(self, x):
        sigmoid = tf.math.sigmoid(x)
        length = self._max_val - self._min_val
        return self._min_val + length * sigmoid
