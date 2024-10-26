import tensorflow as tf


class RecordFieldType:
    STRING = 0
    INT = 1
    FLOAT = 2

    def __init__(self, type: int):
        self._type = type

    def get_tf_type(self):
        if self._type == self.STRING:
            return tf.constant([''], dtype=tf.string)
        if self._type == self.INT:
            return tf.constant([0])
        if self._type == self.FLOAT:
            return tf.constant([0.0])
        raise Exception('The property "type" is invalid, probably a dev issue.')