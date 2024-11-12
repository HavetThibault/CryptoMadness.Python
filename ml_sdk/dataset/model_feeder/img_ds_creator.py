from typing import Any

import pandas as pd
import tensorflow as tf

from ds_creator import DsCreator


def load_grey_img(file_path, width, height):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=1)
    # Tensor of shape (width, height, 1)
    return tf.image.resize(img, [height, width])


def load_rgb_img(file_path, width, height):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    # Tensor of shape (width, height, 3)
    return tf.image.resize(img, [height, width])


class ImgDsCreator(DsCreator):
    def __init__(self, img_dir: str, train_csv_path: str, val_csv_path: str, img_dimensions: list[tuple[int, int]],
                 file_record_struct, repeat_ds: int, batch_size: int, get_img_names, use_cache=False, grey_img=True,
                 get_origin_img_name=None, additional_inputs=0):
        super(ImgDsCreator, self).__init__(
            train_csv_path,
            val_csv_path,
            file_record_struct,
            repeat_ds,
            batch_size,
            use_cache)
        self._img_dir = img_dir
        # img_dimensions = list of (height, width)
        self._img_dimensions: list[tuple[int, int]] = img_dimensions
        self._grey_img = grey_img
        self._get_origin_img_name = get_origin_img_name
        self._additional_inputs = additional_inputs
        self._get_img_names = get_img_names

    def get_additional_inputs(self):
        return self._additional_inputs

    def get_img_nbr(self):
        return len(self._img_dimensions)

    def _get_img_names(self, line_record):
        fields = tf.io.decode_csv(line_record, self._file_record_struct)
        return str(fields[0])

    def _load_dataset_img_names(self, ds_path: str, verbose=False) -> list[str]:
        df = self._get_pandas_df(ds_path)
        img_names = df[df.columns[0]].values
        if verbose:
            print(img_names)
        if self._get_origin_img_name is not None:
            str_img_names = []
            for img_name in img_names:
                str_img_names.append(self._get_origin_img_name(img_name))
            return str_img_names
        return img_names

    @staticmethod
    def static_get_sorted_ds_headers(headers, additional_inputs) -> tuple[str, list[str], list[str]]:
        img_col = headers[0]
        if additional_inputs > 0:
            output_offset = additional_inputs+1
            input_cols = headers[1:output_offset]
        else:
            output_offset = 1
            input_cols = []
        output_cols = headers[output_offset:]
        return img_col, input_cols, output_cols

    def get_sorted_ds_headers(self) -> tuple[str, list[str], list[str]]:
        return self.static_get_sorted_ds_headers(self._ds_headers, self._additional_inputs)

    def get_train_img_names(self, verbose=False) -> list[str]:
        return self._load_dataset_img_names(self._train_csv_path, verbose)

    def get_val_img_names(self, verbose=False) -> list[str]:
        return self._load_dataset_img_names(self._val_csv_path, verbose)

    def _load_images_tensors(self, img_name) -> dict[str, Any]:
        img_tensors = {}
        img_names = self._get_img_names(img_name)
        for k, img_name in enumerate(img_names):
            img_height, img_width = self._img_dimensions[k]
            if self._grey_img:
                img = load_grey_img(self._img_dir + img_name, img_width, img_height)
            else:
                img = load_rgb_img(self._img_dir + img_name, img_width, img_height)
            img_tensors[f'img{k}'] = tf.convert_to_tensor(img)
        return img_tensors
