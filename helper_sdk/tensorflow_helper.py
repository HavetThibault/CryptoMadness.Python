from typing import Any

import tensorflow as tf


def nest_tensor(tensor):
    img_shape_array = list(tensor.shape)
    img_shape_array.insert(0, 1)
    img_new_shape = tuple(img_shape_array)
    return tf.reshape(tensor, img_new_shape)


def get_tensor_filename_ext(img_name) -> tuple[Any, Any]:
    split_tensor = tf.strings.split(img_name, '.')
    base_name_tensor = tf.strings.reduce_join(split_tensor[:-1], separator='.')
    extension_tensor = split_tensor[-1]
    return base_name_tensor, extension_tensor
